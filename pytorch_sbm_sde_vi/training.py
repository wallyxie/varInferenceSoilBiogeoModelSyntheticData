#Python-related imports
import math
from tqdm import tqdm
from typing import Dict, Tuple, Union

#Torch-related imports
import torch
from torch.autograd import Function
from torch import nn
import torch.distributions as D
import torch.nn.functional as F
import torch.optim as optim

#Module imports
from mean_field import *
from obs_and_flow import *
from SBM_SDE_classes import *

'''
This module containins the `calc_log_lik` and `training` functions for pre-training and ELBO training of the soil biogeochemical model SDE systems.
'''

DictOfTensors = Dict[str, torch.Tensor]
Number = Union[int, float]
TupleOfTensors = Tuple[torch.Tensor, torch.Tensor]

###############################
##TRAINING AND ELBO FUNCTIONS##
###############################

def calc_log_lik(C_PATH: torch.Tensor, 
        T_SPAN_TENSOR: torch.Tensor, 
        DT: float, 
        SBM_SDE_CLASS, 
        INIT_PRIOR, 
        PARAMS_DICT: DictOfTensors, 
        LEARN_CO2 = False):

    if LEARN_CO2:
        drift, diffusion_sqrt, x_add_CO2 = SBM_SDE_CLASS.drift_diffusion_add_CO2(C_PATH[:, :-1, :], PARAMS_DICT)
        euler_maruyama_state_sample_object = D.multivariate_normal.MultivariateNormal(loc = C_PATH[:, :-1, :] + drift * DT, scale_tril = diffusion_sqrt * math.sqrt(DT)) #C_PATH[:, :-1, :] + drift * DT will diverge from C_PATH if C_PATH values not compatible with x0 and theta. Algorithm aims to minimize gap between computed drift and actual gradient between x_n and x_{n+1}. 
    
        # Compute log p(x|theta) = log p(x|x0, theta) + log p(x0|theta)
        ll = euler_maruyama_state_sample_object.log_prob(C_PATH[:, 1:, :]).sum(-1) # log p(x|x0, theta)
        ll += INIT_PRIOR.log_prob(C_PATH[:, 0, :]) # log p(x0|theta)
    
        return ll, drift, diffusion_sqrt, x_add_CO2

    else:
        drift, diffusion_sqrt = SBM_SDE_CLASS.drift_diffusion(C_PATH[:, :-1, :], PARAMS_DICT)
        euler_maruyama_state_sample_object = D.multivariate_normal.MultivariateNormal(loc = C_PATH[:, :-1, :] + drift * DT, scale_tril = diffusion_sqrt * math.sqrt(DT)) #C_PATH[:, :-1, :] + drift * DT will diverge from C_PATH if C_PATH values not compatible with x0 and theta. Algorithm aims to minimize gap between computed drift and actual gradient between x_n and x_{n+1}. 
    
        # Compute log p(x|theta) = log p(x|x0, theta) + log p(x0|theta)
        ll = euler_maruyama_state_sample_object.log_prob(C_PATH[:, 1:, :]).sum(-1) # log p(x|x0, theta)
        ll += INIT_PRIOR.log_prob(C_PATH[:, 0, :]) # log p(x0|theta)
    
        return ll, drift, diffusion_sqrt

def train(DEVICE, ELBO_LR, NITER, BATCH_SIZE, NUM_LAYERS,
        OBS_CSV_STR, OBS_ERROR_SCALE, T, DT, N,
        T_SPAN_TENSOR, I_S_TENSOR, I_D_TENSOR, TEMP_TENSOR, TEMP_REF,
        SBM_SDE_CLASS: str, DIFFUSION_TYPE: str,
        INIT_PRIOR, PRIOR_DIST_DETAILS_DICT, LEARN_CO2 = False,
        THETA_DIST = None, THETA_POST_DIST = None, THETA_POST_INIT = None,
        LR_DECAY = 0.8, DECAY_STEP_SIZE = 50000, PRINT_EVERY = 100):

    # if PRETRAIN_ITER >= NITER:
    #     raise ValueError('PRETRAIN_ITER must be < NITER.')

    #Instantiate SBM_SDE object based on specified model and diffusion type.
    SBM_SDE_class_dict = {
            'SCON': SCON,
            'SAWB': SAWB,
            'SAWB-ECA': SAWB_ECA
            }
    if SBM_SDE_CLASS not in SBM_SDE_class_dict:
        raise NotImplementedError('Other SBM SDEs aside from SCON, SAWB, and SAWB-ECA have not been implemented yet.')
    SBM_SDE_class = SBM_SDE_class_dict[SBM_SDE_CLASS]
    SBM_SDE = SBM_SDE_class(T_SPAN_TENSOR, I_S_TENSOR, I_D_TENSOR, TEMP_TENSOR, TEMP_REF, DIFFUSION_TYPE)

    #Read in data to obtain y and establish observation model.
    obs_dim = None
    if LEARN_CO2:
        obs_dim = SBM_SDE.state_dim + 1
    else:
        obs_dim = SBM_SDE.state_dim
    obs_times, obs_means, obs_error = csv_to_obs_df(OBS_CSV_STR, obs_dim, T, OBS_ERROR_SCALE) #csv_to_obs_df function in obs_and_flow module
    obs_model = ObsModel(DEVICE, TIMES = obs_times, DT = DT, MU = obs_means, SCALE = obs_error).to(DEVICE)

    #Establish neural network.
    net = SDEFlow(DEVICE, obs_model, SBM_SDE.state_dim, T, DT, N, num_layers = NUM_LAYERS).to(DEVICE)

    param_names = list(PRIOR_DIST_DETAILS_DICT.keys())

    #Convert prior details dictionary values to tensors.
    prior_list = list(zip(*(PRIOR_DIST_DETAILS_DICT[k] for k in param_names))) #Unzip prior distribution details from dictionary values into individual lists.
    prior_means_tensor, prior_sds_tensor, prior_lowers_tensor, prior_uppers_tensor = torch.tensor(prior_list).to(DEVICE) #Ensure conversion of lists into tensors.

    #Retrieve desired distribution class based on string.
    dist_class_dict = {
            'TruncatedNormal': TruncatedNormal,
            'RescaledLogitNormal': RescaledLogitNormal,
            'MultivariateLogitNormal': MultivariateLogitNormal
            }
    THETA_PRIOR_CLASS = dist_class_dict[THETA_DIST]
    THETA_POST_CLASS = dist_class_dict[THETA_POST_DIST] if THETA_POST_DIST else dist_class_dict[THETA_DIST]
    
    #Define prior
    priors = THETA_PRIOR_CLASS(loc = prior_means_tensor, scale = prior_sds_tensor, a = prior_lowers_tensor, b = prior_uppers_tensor)

    # Initialize posterior q(theta) using its prior p(theta)
    learn_cov = (THETA_POST_DIST == 'MultivariateLogitNormal')
    if THETA_POST_INIT is None:
        THETA_POST_INIT = PRIOR_DIST_DETAILS_DICT
    q_theta = MeanField(DEVICE, param_names, THETA_POST_INIT, THETA_POST_CLASS, learn_cov)

    #Record loss throughout training
    best_loss_norm = 1e20
    best_loss_ELBO = 1e20
    norm_losses = []
    ELBO_losses = []

    #Initiate optimizers.
    pretrain_optimizer = optim.Adam(net.parameters(), lr = PRETRAIN_LR)
    ELBO_params = list(net.parameters()) + list(q_theta.parameters())
    ELBO_optimizer = optim.Adamax(ELBO_params, lr = ELBO_LR)
    
    #Training loop
    with tqdm(total = NITER, desc = f'Train Diffusion', position = -1) as tq:
        for it in range(NITER):
            net.train()
            C_PATH, log_prob = net(BATCH_SIZE) #Obtain paths with solutions to times including t0.
            
            if torch.isnan(C_PATH).any():
                raise ValueError(f'nan in x at niter: {it}. Try reducing learning rate to start.')
            
            # if it <= PRETRAIN_ITER:
            #     pretrain_optimizer.zero_grad()
            #
            #     l1_norm_element = C_PATH - torch.mean(obs_model.mu, -1)[None, None, :] #Compute difference between x and observed state means.
            #     l1_norm = torch.sum(torch.abs(l1_norm_element), (-1, -2)).mean() #Take L1 mean across all samples.
            #     best_loss_norm = l1_norm if l1_norm < best_loss_norm else best_loss_norm
            #     norm_losses.append(l1_norm.item())
            #     #l2_norm_element = C_PATH - torch.mean(obs_model.mu, -1)[None, None, :] #Compute difference between x and observed state means.
            #     #l2_norm = torch.sqrt(torch.sum(torch.square(l2_norm_element), (-1, -2)).mean() #Take L2 mean across all samples.
            #     #best_loss_norm = l2_norm if l2_norm < best_loss_norm else best_loss_norm
            #     #norm_losses.append(l2_norm.item())
            #     
            #     if (it + 1) % PRINT_EVERY == 0:
            #         print(f'Moving average norm loss at {it + 1} iterations is: {sum(norm_losses[-10:]) / len(norm_losses[-10:])}. Best norm loss value is: {best_loss_norm}.')
            #         print('\nC_PATH mean =', C_PATH.mean(-2))
            #         print('\nC_PATH =', C_PATH)
            #
            #     l1_norm.backward()
            #     #l2_norm.backward()
            #     torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)                                
            #     pretrain_optimizer.step()

            # else:
            ELBO_optimizer.zero_grad()                
            
            list_theta = []
            list_parent_loc_scale = []
            theta_dict = None #Initiate theta_dict variable for loop operations.
            theta = None #Initiate theta variable for loop operations.
            log_q_theta = None #Initiate log_q_theta variable for loop operations.
            parent_loc_scale_dict = None #Initiate parent_loc_scale_dict variable for loop operations.
            ELBO = None #Initiate ELBO variable.

            theta_dict, theta, log_q_theta, parent_loc_scale_dict = q_theta(BATCH_SIZE)
            log_p_theta = priors.log_prob(theta).sum(-1)
            list_parent_loc_scale.append(parent_loc_scale_dict)

            if LEARN_CO2:
                log_lik, drift, diffusion_sqrt, x_add_CO2 = calc_log_lik(C_PATH, T_SPAN_TENSOR.to(DEVICE), DT, 
                        I_S_TENSOR.to(DEVICE), I_D_TENSOR.to(DEVICE), TEMP_TENSOR, TEMP_REF, 
                        SBM_SDE, INIT_PRIOR, theta_dict, LEARN_CO2)
                ELBO = -log_p_theta.mean() + log_q_theta.mean() + log_prob.mean() - log_lik.mean() - obs_model(x_add_CO2, theta_dict)
            else:
                log_lik, drift, diffusion_sqrt = calc_log_lik(C_PATH, T_SPAN_TENSOR.to(DEVICE), DT, 
                        I_S_TENSOR.to(DEVICE), I_D_TENSOR.to(DEVICE), TEMP_TENSOR, TEMP_REF, 
                        SBM_SDE, INIT_PRIOR, theta_dict, LEARN_CO2)
                ELBO = -log_p_theta.mean() + log_q_theta.mean() + log_prob.mean() - log_lik.mean() - obs_model(C_PATH, theta_dict)                    
            
            #Negative ELBO: -log p(theta) + log q(theta) - log p(y_0|x_0, theta) [already accounted for in obs_model output when learning x_0] + log q(x|theta) - log p(x|theta) - log p(y|x, theta)
            best_loss_ELBO = ELBO if ELBO < best_loss_ELBO else best_loss_ELBO
            ELBO_losses.append(ELBO.item())

            if (it + 1) % PRINT_EVERY == 0:
                #print('log_prob.mean() =', log_prob.mean())
                #print('log_lik.mean() =', log_lik.mean())
                #print('obs_model(C_PATH, theta_dict) =', obs_model(C_PATH, theta_dict))                    
                #print('drift = ', drift)
                #print('diffusion_sqrt = ', diffusion_sqrt)
                print(f'\nMoving average ELBO loss at {it + 1} iterations is: {sum(ELBO_losses[-10:]) / len(ELBO_losses[-10:])}. Best ELBO loss value is: {best_loss_ELBO}.')
                if not LEARN_CO2:
                    print('\nC_PATH mean =', C_PATH.mean(-2))
                    print('\nC_PATH =', C_PATH)
                if LEARN_CO2:
                    print('\nC_PATH with CO2 mean =', x_add_CO2.mean(-2))
                    print('\nC_PATH with CO2 =', x_add_CO2)
                print('\ntheta_dict means: ', {key: theta_dict[key].mean() for key in param_names})
                print('\nparent_loc_scale_dict: ', parent_loc_scale_dict)

            ELBO.backward()
            torch.nn.utils.clip_grad_norm_(ELBO_params, 5.0)
            ELBO_optimizer.step()
        
            if it % DECAY_STEP_SIZE == 0:
                ELBO_optimizer.param_groups[0]['lr'] *= LR_DECAY

        tq.update()
    
    return net, q_theta, priors, obs_model, ELBO_losses, list_parent_loc_scale    
