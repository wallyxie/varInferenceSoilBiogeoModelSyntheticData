import math

from tqdm import tqdm

from obs_and_flow import *
from mean_field import *
from mean_field_tmp import *

import torch
from torch.autograd import Function
from torch import nn
import torch.distributions as D
import torch.nn.functional as F
import torch.optim as optim

'''
This module containins the `calc_log_lik` and `training` functions for pre-training and ELBO training of the soil biogeochemical model SDE systems.
'''

###############################
##TRAINING AND ELBO FUNCTIONS##
###############################

def calc_log_lik(C_PATH, T_SPAN_TENSOR, DT, I_S_TENSOR, I_D_TENSOR, TEMP_TENSOR, TEMP_REF, DRIFT_DIFFUSION, INIT_PRIOR, PARAMS_DICT):
    drift, diffusion_sqrt = DRIFT_DIFFUSION(C_PATH[:, :-1, :], T_SPAN_TENSOR[:, :-1, :], I_S_TENSOR[:, :-1, :], I_D_TENSOR[:, :-1, :], TEMP_TENSOR[:, :-1, :], TEMP_REF, PARAMS_DICT)
    #print('\ndrift = ', drift)
    #print('\ndiffusion = ', diffusion_sqrt)      
    euler_maruyama_state_sample_object = D.multivariate_normal.MultivariateNormal(loc = C_PATH[:, :-1, :] + drift * DT, scale_tril = diffusion_sqrt * math.sqrt(DT))
    
    # Compute log p(x|theta) = log p(x|x0, theta) + log p(x0|theta)
    ll = euler_maruyama_state_sample_object.log_prob(C_PATH[:, 1:, :]).sum(-1) # log p(x|x0, theta)
    ll += INIT_PRIOR.log_prob(C_PATH[:, 0, :]) # log p(x0|theta)
    
    return ll, drift, diffusion_sqrt

def train(DEVICE, PRETRAIN_LR, ELBO_LR, NITER, PRETRAIN_ITER, BATCH_SIZE, NUM_LAYERS,
          STATE_DIM, OBS_CSV_STR, OBS_ERROR_SCALE, T, DT, N, T_SPAN_TENSOR, I_S_TENSOR, I_D_TENSOR, TEMP_TENSOR, TEMP_REF,
          DRIFT_DIFFUSION, INIT_PRIOR, PRIOR_DICT, THETA_DIST = None,
          LEARN_THETA = False, LR_DECAY = 0.9, DECAY_STEP_SIZE = 1000, PRINT_EVERY = 10):
    if PRETRAIN_ITER >= NITER:
        raise ValueError('PRETRAIN_ITER must be < NITER.')

    #Read in data to obtain y and establish observation model.
    obs_times, obs_means_noCO2, obs_error = csv_to_obs_df(OBS_CSV_STR, STATE_DIM, T, OBS_ERROR_SCALE) #csv_to_obs_df function in obs_and_flow module
    obs_model = ObsModel(DEVICE, TIMES = obs_times, DT = DT, MU = obs_means_noCO2, SCALE = obs_error).to(DEVICE) 

    #Establish neural network.
    #net = SDEFlow(DEVICE, obs_model, STATE_DIM, T, DT, N, I_S_TENSOR = I_S_TENSOR, I_D_TENSOR = I_D_TENSOR, num_layers = NUM_LAYERS).to(DEVICE)
    net = SDEFlow(DEVICE, obs_model, STATE_DIM, T, DT, N, num_layers = NUM_LAYERS).to(DEVICE)
    
    if LEARN_THETA:
        # Ensure consistent order b/w prior p and variational posterior q
        param_names = list(PRIOR_DICT.keys())

        #Convert prior details dictionary values to tensors.
        prior_list = list(zip(*(PRIOR_DICT[k] for k in param_names))) #Unzip prior distribution details from dictionary values into individual lists.
        prior_means_tensor, prior_sds_tensor, prior_lowers_tensor, prior_uppers_tensor = torch.tensor(prior_list).to(DEVICE) #Ensure conversion of lists into tensors.

        # Define prior
        priors = THETA_DIST(loc = prior_means_tensor, scale = prior_sds_tensor, a = prior_lowers_tensor, b = prior_uppers_tensor)
        #priors1 = BoundedNormal(DEVICE, param_names, PRIOR_DICT)

        # Initialize posterior q(theta) using its prior p(theta)
        q_theta = MeanField(DEVICE, param_names, PRIOR_DICT, THETA_DIST)
        #q_theta1 = MeanFieldTmp(DEVICE, param_names, PRIOR_DICT)
    else:
        #Establish initial dictionary of theta means in tensor form.
        theta_dict = {k: torch.tensor(v).to(DEVICE).expand(BATCH_SIZE) for k, (v, _, _) in PRIOR_DICT.items()}
        q_theta = None

    #Record loss throughout training
    best_loss_norm = 1e20
    best_loss_ELBO = 1e20
    norm_losses = []
    ELBO_losses = []

    #Initiate optimizers.
    pretrain_optimizer = optim.Adam(net.parameters(), lr = PRETRAIN_LR)
    if LEARN_THETA:
        ELBO_params = list(net.parameters()) + list(q_theta.parameters())
        ELBO_optimizer = optim.Adam(ELBO_params, lr = ELBO_LR)
    else:
        ELBO_optimizer = optim.Adam(net.parameters(), lr = ELBO_LR)
    
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
                
                #Commented out because log p(y_0 | x_0, theta) already accounted for in output of obs_model (we are now learning x_0).
                #Create x_0 prior
                #x_0 = C_PATH[:, 0, :]
                #x_0_prior = D.normal.Normal(loc = x_0, scale = x_0 * OBS_ERROR_SCALE)
                #log p(y_0 | x_0, theta)
                #log_p_y_0 = x_0_prior.log_prob(obs_model.mu[:, 0]).sum(-1)

                list_theta = []
                list_parent_loc_scale = []
                theta_dict = None #Initiate theta_dict variable for printing in PRINT_EVERY loop.
                parent_loc_scale_dict = None #Initiate parent_loc_scale_dict variable for printing in PRINT_EVERY loop.

                if LEARN_THETA:
                    theta_dict, theta, log_q_theta = q_theta(BATCH_SIZE)
                    #theta_dict1, theta1, log_q_theta1 = q_theta1(BATCH_SIZE)
                    
                    log_p_theta = priors.log_prob(theta).sum(-1)
                    #log_p_theta1 = priors.log_prob(theta1).sum(-1)
                    #print(it, log_q_theta, log_p_theta)
                    #list_theta.append(theta_dict)
                    #list_parent_loc_scale.append(parent_loc_scale_dict)
                else:
                    log_q_theta, log_p_theta = torch.zeros(2).to(DEVICE)

                log_lik, drift, diffusion_sqrt = calc_log_lik(C_PATH, T_SPAN_TENSOR.to(DEVICE), DT, I_S_TENSOR.to(DEVICE), I_D_TENSOR.to(DEVICE),
                                       TEMP_TENSOR, TEMP_REF, DRIFT_DIFFUSION, INIT_PRIOR, theta_dict)
                
                #Negative ELBO: -log p(theta) + log q(theta) - log p(y_0|x_0, theta) [already accounted for in obs_model output when learning x_0] + log q(x|theta) - log p(x|theta) - log p(y|x, theta)
                ELBO = -log_p_theta.mean() + log_q_theta.mean() + log_prob.mean() - log_lik.mean() - obs_model(C_PATH, theta_dict)
                best_loss_ELBO = ELBO if ELBO < best_loss_ELBO else best_loss_ELBO
                ELBO_losses.append(ELBO.item())

                if (it + 1) % PRINT_EVERY == 0:
                    #print('log_prob.mean() =', log_prob.mean())
                    #print('log_lik.mean() =', log_lik.mean())
                    #print('obs_model(C_PATH, theta_dict) =', obs_model(C_PATH, theta_dict))                    
                    #print('drift = ', drift)
                    #print('diffusion_sqrt = ', diffusion_sqrt)
                    print('\ntheta_dict = ', theta_dict)
                    print('\nparent_loc_scale_dict = ', parent_loc_scale_dict)
                    print(f'\nMoving average ELBO loss at {it + 1} iterations is: {sum(ELBO_losses[-10:]) / len(ELBO_losses[-10:])}. Best ELBO loss value is: {best_loss_ELBO}.')
                    print('\nC_PATH mean =', C_PATH.mean(-2))
                    print('\nC_PATH =', C_PATH)

                    if LEARN_THETA:
                        print('\ntheta_dict = ', {key: theta_dict[key].mean() for key in param_names})

                ELBO.backward()
                if LEARN_THETA:
                    torch.nn.utils.clip_grad_norm_(ELBO_params, 5.0)
                else:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
                ELBO_optimizer.step()
            
                if it % DECAY_STEP_SIZE == 0:
                    ELBO_optimizer.param_groups[0]['lr'] *= LR_DECAY

            tq.update()
            
    return net, q_theta, obs_model, ELBO_losses #, list_parent_loc_scale
