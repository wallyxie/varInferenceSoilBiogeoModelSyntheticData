from obs_and_flow import *
from mean_field import *
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

def calc_log_lik(C_PATH, T_SPAN_TENSOR, DT, I_S_TENSOR, I_D_TENSOR, TEMP_TENSOR, TEMP_REF, DRIFT_DIFFUSION, X0_PRIOR, PARAMS_DICT):
    drift, diffusion_sqrt = DRIFT_DIFFUSION(C_PATH[:, :-1, :], T_SPAN_TENSOR[:, :-1, :], I_S_TENSOR[:, :-1, :], I_D_TENSOR[:, :-1, :], TEMP_TENSOR[:, :-1, :], TEMP_REF, PARAMS_DICT)
    #print('\ndrift = ', drift)
    #print('\ndiffusion = ', diffusion_sqrt)      
    euler_maruyama_state_sample_object = D.multivariate_normal.MultivariateNormal(loc = C_PATH[:, :-1, :] + drift * DT, scale_tril = diffusion_sqrt * math.sqrt(DT))
    
    # Compute log p(x|theta) = log p(x|x0, theta) + log p(x0|theta)
    ll = euler_maruyama_state_sample_object.log_prob(C_PATH[:, 1:, :]).sum(-1) # log p(x|x0, theta)
    ll += X0_PRIOR.log_prob(C_PATH[:, 0, :]) # log p(x0|theta)
    
    return ll

def train(DEVICE, PRETRAIN_LR, ELBO_LR, NITER, PRETRAIN_ITER, BATCH_SIZE, NUM_LAYERS,
          STATE_DIM, OBS_CSV_STR, OBS_ERROR_SCALE, PRIOR_SCALE_FACTOR, T, DT, N, T_SPAN_TENSOR, I_S_TENSOR, I_D_TENSOR, TEMP_TENSOR, TEMP_REF,
          DRIFT_DIFFUSION, X0_PRIOR, PARAM_PRIOR_MEANS_DICT,
          LEARN_THETA = False, LR_DECAY = 0.9, DECAY_STEP_SIZE = 1000, PRINT_EVERY = 10):

    if PRETRAIN_ITER >= NITER:
        raise ValueError('PRETRAIN_ITER must be < NITER.')

    #Convert prior means dictionary values to tensor.
    prior_means_tensor = torch.Tensor(list(PARAM_PRIOR_MEANS_DICT.values())).to(DEVICE)

    #Establish initial dictionary of theta means in tensor form.
    theta_dict = {k: torch.tensor(v).to(DEVICE).expand(BATCH_SIZE) for k, v in PARAM_PRIOR_MEANS_DICT.items()}

    #Read in data to obtain y and establish observation model.
    obs_times, obs_means_noCO2, obs_error = csv_to_obs_df(OBS_CSV_STR, STATE_DIM, T, OBS_ERROR_SCALE) #csv_to_obs_df function in obs_and_flow module
    obs_model = ObsModel(DEVICE, TIMES = obs_times, DT = DT, MU = obs_means_noCO2, SCALE = obs_error).to(DEVICE) 

    #Establish neural network.
    #net = SDEFlow(DEVICE, obs_model, STATE_DIM, T, DT, N, I_S_TENSOR = I_S_TENSOR, I_D_TENSOR = I_D_TENSOR, num_layers = NUM_LAYERS).to(DEVICE)
    net = SDEFlow(DEVICE, obs_model, STATE_DIM, T, DT, N, num_layers = NUM_LAYERS).to(DEVICE)
    
    if LEARN_THETA:
        priors = D.normal.Normal(prior_means_tensor, prior_means_tensor * PRIOR_SCALE_FACTOR)
        q_theta = MeanField(PARAM_PRIOR_MEANS_DICT, PRIOR_SCALE_FACTOR)
    else:
        q_theta = None

    #Record loss throughout training
    best_loss_norm = 1e20
    best_loss_ELBO = 1e20
    norm_losses = []
    ELBO_losses = []

    #Initiate optimizers.
    pretrain_optimizer = optim.Adam(net.parameters(), lr = PRETRAIN_LR)
    ELBO_params = list(net.parameters()) + list(q_theta.parameters()) if LEARN_THETA else net.parameters()
    ELBO_optimizer = optim.Adam(ELBO_params, lr = ELBO_LR)

    #C0 = ANALYTICAL_STEADY_STATE_INIT(I_S_TENSOR[0, 0, 0].item(), I_D_TENSOR[0, 0, 0].item(), PARAM_PRIOR_MEANS_DICT) #Calculate deterministic initial conditions.
    #C0 = C0[(None,) * 2].repeat(BATCH_SIZE, 1, 1).to(DEVICE) #Assign initial conditions to C_PATH.
    
    #Training loop
    with tqdm(total = NITER, desc = f'Train Diffusion', position = -1) as tq:
        for it in range(NITER):
            net.train()
            C_PATH, log_prob = net(BATCH_SIZE) #Obtain paths with solutions to times including t0.
            #C_PATH = torch.cat([C0, C_PATH], 1) #Append deterministic CON initial conditions conditional on parameter values to C path.
            
            if torch.isnan(C_PATH).any():
                raise ValueError(f'nan in x at niter: {it}. Try reducing learning rate to start.')
            
            if it <= PRETRAIN_ITER:
                pretrain_optimizer.zero_grad()

                l1_norm_element = C_PATH - torch.mean(obs_model.mu, -1)[None, None, :] #Compute difference between x and observed state means.
                l1_norm = torch.sum(torch.abs(l1_norm_element), (-1, -2)).mean() #Take L1 mean across all samples.
                best_loss_norm = l1_norm if l1_norm < best_loss_norm else best_loss_norm
                norm_losses.append(l1_norm.item())
                #l2_norm_element = C_PATH - torch.mean(obs_model.mu, -1)[None, None, :] #Compute difference between x and observed state means.
                #l2_norm = torch.sqrt(torch.sum(torch.square(l2_norm_element), (-1, -2)).mean() #Take L2 mean across all samples.
                #best_loss_norm = l2_norm if l2_norm < best_loss_norm else best_loss_norm
                #norm_losses.append(l2_norm.item())
                
                if (it + 1) % PRINT_EVERY == 0:
                    print(f'Moving average norm loss at {it + 1} iterations is: {sum(norm_losses[-10:]) / len(norm_losses[-10:])}. Best norm loss value is: {best_loss_norm}.')
                    print('\nC_PATH mean =', C_PATH.mean(-2))
                    print('\nC_PATH =', C_PATH)

                l1_norm.backward()
                #l2_norm.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)                                
                pretrain_optimizer.step()

            else:
                ELBO_optimizer.zero_grad()                
                
                if LEARN_THETA:
                    theta_dict, theta, log_q_theta = q_theta(BATCH_SIZE)
                    print('\ntheta_dict = ', theta_dict)
                    log_p_theta = priors.log_prob(theta).sum(-1)
                    #log_p_y_0_giv_x_0_and_theta = 0 #Temporary value. log_p_y_0_giv_x_0_and_theta = obs_model(C_0, theta_dict)/obs_model_CO2(C_0_with_CO2, theta_dict)
                else:
                    log_q_theta, log_p_theta = torch.zeros(2).to(DEVICE)                    
                    #log_p_y_0_giv_x_0_and_theta = 0 #Ignore for now. Presently, no separate probability modeled for y_0 given x_0.

                neg_log_lik = calc_log_lik(C_PATH, T_SPAN_TENSOR.to(DEVICE), DT, I_S_TENSOR.to(DEVICE), I_D_TENSOR.to(DEVICE),
                                       TEMP_TENSOR, TEMP_REF, DRIFT_DIFFUSION, X0_PRIOR, theta_dict)
                
                # - log p(theta) + log q(theta) + log q(x|theta) - log p(x|theta) - log p(y|x, theta)
                ELBO = -log_p_theta.mean() + log_q_theta.mean() + log_prob.mean() + neg_log_lik.mean() - obs_model(C_PATH, theta_dict)
                best_loss_ELBO = ELBO if ELBO < best_loss_ELBO else best_loss_ELBO
                ELBO_losses.append(ELBO.item())

                if (it + 1) % PRINT_EVERY == 0:
                    print('log_prob.mean() =', log_prob.mean())
                    print('log_lik.mean() =', neg_log_lik.mean())
                    print('obs_model(C_PATH, theta_dict) =', obs_model(C_PATH, theta_dict))
                    print(f'Moving average ELBO loss at {it + 1} iterations is: {sum(ELBO_losses[-10:]) / len(ELBO_losses[-10:])}. Best ELBO loss value is: {best_loss_ELBO}.')
                    print('\nC_PATH mean =', C_PATH.mean(-2))
                    print('\n C_PATH =', C_PATH)

                ELBO.backward()
                torch.nn.utils.clip_grad_norm_(ELBO_params, 3.0)                
                ELBO_optimizer.step()                
            
                if it % DECAY_STEP_SIZE == 0:
                    ELBO_optimizer.param_groups[0]['lr'] *= LR_DECAY

            tq.update()
            
    return net, ELBO_losses
