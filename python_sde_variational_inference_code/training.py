from obs_and_flow import *
import torch
from torch.autograd import Function
from torch import nn
import torch.distributions as D
import torch.nn.functional as F
import torch.optim as optim
import math
from tqdm import tqdm

'''
This module containins the `calc_log_lik` and `training` functions for pre-training and ELBO training of the soil biogeochemical model SDE systems.
'''

###############################
##TRAINING AND ELBO FUNCTIONS##
###############################

def calc_log_lik(C_PATH, T_SPAN_TENSOR, DT, I_S_TENSOR, I_D_TENSOR, DRIFT_DIFFUSION, PARAMS_DICT, TEMP_GEN, TEMP_REF):
    drift, diffusion_sqrt = DRIFT_DIFFUSION(C_PATH[:, :-1, :], T_SPAN_TENSOR[:, :-1, :], I_S_TENSOR[:, :-1, :], I_D_TENSOR[:, :-1, :], PARAMS_DICT, TEMP_GEN, TEMP_REF)
    euler_maruyama_state_sample_object =D.multivariate_normal.MultivariateNormal(loc = C_PATH[:, :-1, :] + drift * DT, scale_tril = diffusion_sqrt * math.sqrt(DT))
    return euler_maruyama_state_sample_object.log_prob(C_PATH[:, 1:, :]).sum(-1)

def train(DEVICE, PRETRAIN_LR, ELBO_LR, NITER, PRETRAIN_ITER, BATCH_SIZE,
          PRIOR_SCALE_FACTOR, SDEFLOW, ObsModel, csv_to_obs_df, DATA_CSV,
          OBS_ERROR_SCALE_FACTOR, STATE_DIM, T, DT, N, T_SPAN_TENSOR,
          I_S_TENSOR, I_D_TENSOR, DRIFT_DIFFUSION, PARAM_PRIOR_MEANS_DICT,
          TEMP_GEN, TEMP_REF, ANALYTICAL_STEADY_STATE_INIT,
          LR_DECAY = 0.1, DECAY_STEP_SIZE = 100000, LEARN_THETA = False, PRINT_EVERY = 10):
    if PRETRAIN_ITER >= NITER:
        raise Exception("PRETRAIN_ITER must be < NITER.")
    
    # Load data
    obs_times, obs_means, obs_error = csv_to_obs_df(DATA_CSV, STATE_DIM + 1, T, OBS_ERROR_SCALE_FACTOR) #Import data CSV and extract observation times, means, and desired observation error standard deviation based on `obs_error_scale_factor`. 
    
    # Initialize models
    obs_model = ObsModel(DEVICE, obs_times, DT, obs_means[:-1, :], obs_error[:, :-1]) #Hack for bypassing ObsModel and SDEFlow dimension mismatch issue.
    net = SDEFLOW(DEVICE, obs_model, STATE_DIM, T, DT, N,
                  I_S_TENSOR, I_D_TENSOR, cond_inputs = 3).to(DEVICE) #Instantiate flow.
    prior_means_tensor = torch.Tensor(list(PARAM_PRIOR_MEANS_DICT.values())) #Convert prior mean dictionary values to tensor.
    
    #Establish approximate priors.
    theta_dict = {k: torch.tensor(v).expand(BATCH_SIZE) for k, v in PARAM_PRIOR_MEANS_DICT.items()}
    
    if LEARN_THETA:
        priors = D.normal.Normal(prior_means_tensor, prior_means_tensor * PRIOR_SCALE_FACTOR)
        q_theta = MeanField(PARAM_PRIOR_MEANS_DICT, PRIOR_SCALE_FACTOR)
    else:
        q_theta = None
    
    #Initiate optimizers.
    pretrain_optimizer = optim.Adam(net.parameters(), lr = PRETRAIN_LR, eps = 1e-7)
    elbo_params = list(net.parameters()) + list(q_theta.parameters()) if LEARN_THETA \
        else net.parameters()
    ELBO_optimizer = optim.Adamax(elbo_params, lr = ELBO_LR)
    
    # Record loss throughout training
    best_loss_norm = 1e10
    best_loss_ELBO = 1e20
    norm_losses = [] #[best_loss_norm] * 10
    ELBO_losses = [] #[best_loss_ELBO] * 10
    
    #Joint optimization of SDE and hidden (NN) parameters loop.
    with tqdm(total = NITER, desc = f'\nTrain Diffusion', position = -1) as tq:
        for it in range(NITER):
            net.train()
            C_PATH, log_prob = net(BATCH_SIZE) #Obtain paths with solutions at times after t0.
            
            #Control flow for LEARN_THETA setting.
            if LEARN_THETA:
                theta_dict, theta, log_q_theta = q_theta(BATCH_SIZE)
                print('\ntheta_dict = ', theta_dict)
                log_p_theta = priors.log_prob(theta).sum(-1)
                log_p_y_0_giv_x_0_and_theta = 0 #Temporary. log_p_y_0_giv_x_0_and_theta = obs_model(C_0, theta_dict)/obs_model_CO2(C_0_with_CO2, theta_dict)
            else:
                log_q_theta = torch.tensor(0.)
                log_p_theta = torch.tensor(0.)
                log_p_y_0_giv_x_0_and_theta = 0
            
            #Initial C conditions estimation based on theta.
            C_0 = LowerBound.apply(ANALYTICAL_STEADY_STATE_INIT(I_S_TENSOR[0, 0, 0].item(), I_D_TENSOR[0, 0, 0].item(), theta_dict), 1e-7) #Calculate deterministic initial conditions.
            print('\nC_0 =', C_0)
            #C0 = C0[(None,) * 2].repeat(BATCH_SIZE, 1, 1).to(DEVICE) #Commenting out because analytical steady state init functions now output tensors with appropriate batch size if argument into MeanField forward function is BATCH_SIZE. #Assign initial conditions to C_PATH.
            C_PATH = torch.cat([C_0.unsqueeze(1), C_PATH], 1) #Append deterministic CON initial conditions conditional on parameter values to C path. 
            print('\nC_PATH =', C_PATH)
            print('\nC_PATH mean =', C_PATH.mean(-2))
            
            if it <= PRETRAIN_ITER:
                pretrain_optimizer.zero_grad()
                #l1_norm_element = C_PATH - torch.mean(obs_model.mu, -1)
                #l1_norm = torch.sum(torch.abs(l1_norm_element)).mean()
                #best_loss_norm = l1_norm if l1_norm < best_loss_norm else best_loss_norm
                #norm_losses.append(l1_norm.item())
                l2_norm_element = C_PATH - torch.mean(obs_model.mu, -1)
                l2_norm = torch.sqrt(torch.sum(torch.square(l2_norm_element))).mean()
                best_loss_norm = l2_norm if l2_norm < best_loss_norm else best_loss_norm
                norm_losses.append(l2_norm.item())
                
                if it % PRINT_EVERY == 0:
                    ma_norm_loss = sum(norm_losses[-10:]) / len(norm_losses[-10:])
                    print(f"\nMoving average norm loss at {iter} iterations is: {ma_norm_loss}. Best norm loss value is: {best_loss_norm}.")
                    print('\nC_PATH mean =', C_PATH.mean(-2))
                    print('\nC_PATH =', C_PATH)
                #l1_norm.backward()
                l2_norm.backward()
                pretrain_optimizer.step()
            else:
                ELBO_optimizer.zero_grad()                
                log_lik = calc_log_lik(C_PATH, T_SPAN_TENSOR.to(DEVICE), DT,
                                       I_S_TENSOR.to(DEVICE), I_D_TENSOR.to(DEVICE),
                                       DRIFT_DIFFUSION, theta_dict, TEMP_GEN, TEMP_REF)
                neg_ELBO = -log_p_theta.mean() + log_q_theta.mean() - log_p_y_0_giv_x_0_and_theta\
                    - log_lik.mean() - obs_model(C_PATH, theta_dict) + log_prob.mean() #From equation 14 of Ryder et al., 2019.
                #neg_ELBO = -log_lik.mean() - obs_model(C_PATH) + log_prob.mean() #Old ELBO computation without joint density optimization.
                print('\nneg_ELBO_mean = ', neg_ELBO)
                best_loss_ELBO = neg_ELBO if neg_ELBO < best_loss_ELBO else best_loss_ELBO
                ELBO_losses.append(neg_ELBO)
                
                if it % PRINT_EVERY == 0:
                    ma_elbo_loss = sum(ELBO_losses[-10:]) / len(ELBO_losses[-10:])
                    print(f"\nMoving average ELBO loss at {iter} iterations is: {ma_elbo_loss}. Best ELBO loss value is: {best_loss_ELBO}.")
                neg_ELBO.backward()
                ELBO_optimizer.step()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
            if it % DECAY_STEP_SIZE == 0 and it > 0:
                ELBO_optimizer.param_groups[0]['lr'] *= LR_DECAY
            tq.update()
            
    return net, q_theta, ELBO_losses, norm_losses
