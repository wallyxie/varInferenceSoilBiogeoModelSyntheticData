#Python-related imports
import math
from tqdm import tqdm
from typing import Dict, Tuple, Union
import platform

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
DictOfNpArrays = Dict[str, np.ndarray]
Number = Union[int, float]
BoolAndString = Union[bool, str]

###############################
##TRAINING AND ELBO FUNCTIONS##
###############################

def calc_log_lik1(C_PATH: torch.Tensor,
        PARAMS_DICT: DictOfTensors,
        DT: float, 
        SBM_SDE_CLASS, 
        INIT_PRIOR,
        ):

    drift, diffusion_sqrt = SBM_SDE_CLASS.drift_diffusion(C_PATH, PARAMS_DICT) #Appropriate indexing of tensors corresponding to data generating process now handled in `drift_diffusion` class method. Recall that drift diffusion will use C_PATH[:, :-1, :], I_S_TENSOR[:, 1:, :], I_D_TENSOR[:, 1:, :], TEMP_TENSOR[:, 1:, :]. 
    euler_maruyama_state_sample_object = D.multivariate_normal.MultivariateNormal(loc = C_PATH[:, :-1, :] + drift * DT, scale_tril = diffusion_sqrt * math.sqrt(DT)) #C_PATH[:, :-1, :] + drift * DT will diverge from C_PATH if C_PATH values not compatible with x0 and theta. Algorithm aims to minimize gap between computed drift and actual gradient between x_n and x_{n+1}. 

    # Compute log p(x|theta) = log p(x|x0, theta) + log p(x0|theta)
    ll = euler_maruyama_state_sample_object.log_prob(C_PATH[:, 1:, :]).sum(-1) # log p(x|x0, theta)
    ll += INIT_PRIOR.log_prob(C_PATH[:, 0, :]) # log p(x0|theta)

    return ll, drift, diffusion_sqrt

def calc_log_lik2(C_PATH: torch.Tensor,
        PARAMS_DICT: DictOfTensors,
        DT: float, 
        SBM_SDE_CLASS, 
        INIT_PRIOR,
        LEARN_CO2
        ):
    if LEARN_CO2:
        drift, diffusion_sqrt, x_add_CO2 = SBM_SDE_CLASS.drift_diffusion_add_CO2(C_PATH, PARAMS_DICT) #Appropriate indexing of tensors corresponding to data generating process now handled in `drift_diffusion` class method. Recall that drift diffusion will use C_PATH[:, :-1, :], I_S_TENSOR[:, 1:, :], I_D_TENSOR[:, 1:, :], TEMP_TENSOR[:, 1:, :]. 
        euler_maruyama_state_sample_object = D.multivariate_normal.MultivariateNormal(loc = C_PATH[:, :-1, :] + drift * DT, scale_tril = diffusion_sqrt * math.sqrt(DT)) #C_PATH[:, :-1, :] + drift * DT will diverge from C_PATH if C_PATH values not compatible with x0 and theta. Algorithm aims to minimize gap between computed drift and actual gradient between x_n and x_{n+1}. 

        # Compute log p(x|theta) = log p(x|x0, theta) + log p(x0|theta)
        ll = euler_maruyama_state_sample_object.log_prob(C_PATH[:, 1:, :]).sum(-1) # log p(x|x0, theta)
        ll += INIT_PRIOR.log_prob(C_PATH[:, 0, :]) # log p(x0|theta)

        return ll, drift, diffusion_sqrt, x_add_CO2
    else:
        drift, diffusion_sqrt = SBM_SDE_CLASS.drift_diffusion(C_PATH, PARAMS_DICT) #Appropriate indexing of tensors corresponding to data generating process now handled in `drift_diffusion` class method. Recall that drift diffusion will use C_PATH[:, :-1, :], I_S_TENSOR[:, 1:, :], I_D_TENSOR[:, 1:, :], TEMP_TENSOR[:, 1:, :]. 
        euler_maruyama_state_sample_object = D.multivariate_normal.MultivariateNormal(loc = C_PATH[:, :-1, :] + drift * DT, scale_tril = diffusion_sqrt * math.sqrt(DT)) #C_PATH[:, :-1, :] + drift * DT will diverge from C_PATH if C_PATH values not compatible with x0 and theta. Algorithm aims to minimize gap between computed drift and actual gradient between x_n and x_{n+1}. 

        # Compute log p(x|theta) = log p(x|x0, theta) + log p(x0|theta)
        ll = euler_maruyama_state_sample_object.log_prob(C_PATH[:, 1:, :]).sum(-1) # log p(x|x0, theta)
        ll += INIT_PRIOR.log_prob(C_PATH[:, 0, :]) # log p(x0|theta)

        return ll, drift, diffusion_sqrt

def train(DEVICE, ELBO_LR: float, ELBO_ITER: int, BATCH_SIZE: int,
        OBS_CSV_STR: str, OBS_ERROR_SCALE: float, T: float, DT: float, N: int,
        T_SPAN_TENSOR: torch.Tensor, I_S_TENSOR: torch.Tensor, I_D_TENSOR: torch.Tensor, TEMP_TENSOR: torch.Tensor, TEMP_REF: float,
        SBM_SDE_CLASS: str, DIFFUSION_TYPE: str, INIT_PRIOR : torch.distributions.distribution.Distribution, 
        PRIOR_DIST_DETAILS_DICT: DictOfTensors, FIX_THETA_DICT = None, LEARN_CO2: bool = False,
        THETA_DIST = None, THETA_POST_DIST = None, THETA_POST_INIT = None,
        BYPASS_NAN: bool = False, ELBO_LR_DECAY: float = 0.8, ELBO_LR_DECAY_STEP_SIZE: int = 50000, PTRAIN_LR_DECAY: float = 0.8, PTRAIN_LR_DECAY_STEP_SIZE: int = 1000,
        PRINT_EVERY: int = 100, DEBUG_SAVE_DIR: str = None, PTRAIN_ITER: int = 0, PTRAIN_LR: float = None, PTRAIN_ALG: BoolAndString = False,
        NUM_LAYERS: int = 5):

    #Sum to get total training iterations.
    T_ITER = ELBO_ITER + PTRAIN_ITER

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
    if LEARN_CO2:
        obs_dim = SBM_SDE.state_dim + 1
    else:
        obs_dim = SBM_SDE.state_dim
    obs_times, obs_means, obs_error = csv_to_obs_df(OBS_CSV_STR, obs_dim, T, OBS_ERROR_SCALE) #csv_to_obs_df function in obs_and_flow module
    obs_model = ObsModel(TIMES = obs_times, DT = DT, MU = obs_means, SCALE = obs_error).to(DEVICE)

    if LEARN_CO2 and PTRAIN_ITER != 0 and PTRAIN_ALG:
        mean_state_obs = torch.mean(obs_model.mu[:-1, :], -1)[None, None, :]
    elif not LEARN_CO2 and PTRAIN_ITER != 0 and PTRAIN_ALG:
        mean_state_obs = torch.mean(obs_model.mu, -1)[None, None, :]

    #Establish neural network.
    net = SDEFlow(DEVICE, obs_model, SBM_SDE.state_dim, T, DT, N, num_layers = NUM_LAYERS).to(DEVICE)

    #Initiate model debugging saver.
    if DEBUG_SAVE_DIR:
        debug_saver = ModelSaver(save_dir = DEBUG_SAVE_DIR)

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
    best_loss_norm = 1e15
    best_loss_ELBO = 1e15
    norm_losses = []
    ELBO_losses = []

    #Initiate optimizers.
    if PTRAIN_ALG and PTRAIN_ITER != 0:
        ptrain_opt = optim.Adam(net.parameters(), lr = PTRAIN_LR)
        if LEARN_CO2:
            mean_state_obs = torch.mean(obs_model.mu[:-1, :], -1)[None, None, :]
        else:
            mean_state_obs = torch.mean(obs_model.mu, -1)[None, None, :]
    elif not PTRAIN_ALG and PTRAIN_ITER != 0:
        raise Error('Pre-training iterations specified without PTRAIN_ALG input. Must request PTRAIN_ALG = "L1" or "L2".')
    ELBO_params = list(net.parameters()) + list(q_theta.parameters())
    ELBO_opt = optim.Adamax(ELBO_params, lr = ELBO_LR)
    
    net.train()    
    #Training loop
    print(f'\nStarting autoencoder training. {PTRAIN_ITER} pre-training iterations and {ELBO_ITER} ELBO training iterations for {T_ITER} total iterations specified.')        
    with tqdm(total = T_ITER, desc = f'Learning SDE and hidden parameters.', position = -1) as tq:
        for it in range(T_ITER):
            C_PATH, log_prob = net(BATCH_SIZE) #Obtain paths with solutions to times including t0.
            
            #NaN handling            
            nan_count = 0
            #Check for NaNs in x.
            if torch.isnan(C_PATH).any():
                if BYPASS_NAN:
                    nan_count += 1
                    print(f'nan_count = {nan_count}')
                    print(f'Warning. NaN in x at niter: {it}. Using `torch.nan_to_num` to bypass. Check gradient clipping and learning rate to start.')
                    C_PATH = torch.nan_to_num(C_PATH)
                else:
                    raise ValueError(f'nan in x at niter: {it}. Check gradient clipping and learning rate to start.')
            
            if PTRAIN_ALG and it < PTRAIN_ITER:
                ptrain_opt.zero_grad()

                if PTRAIN_ALG == 'L1':
                    l1_norm_element = C_PATH - mean_state_obs #Compute difference between x and observed state means.
                    norm = torch.sum(torch.abs(l1_norm_element), (-1, -2)).mean() #Take L1 mean across all samples.
                    best_loss_norm = norm if norm < best_loss_norm else best_loss_norm
                    norm_losses.append(norm.item())

                elif PTRAIN_ALG == 'L2':
                    l2_norm_element = C_PATH - mean_state_obs #Compute difference between x and observed state means.
                    norm = torch.sqrt(torch.sum(torch.square(l2_norm_element), (-1, -2))).mean() #Take L2 mean across all samples.
                    best_loss_norm = norm if norm < best_loss_norm else best_loss_norm
                    norm_losses.append(norm.item())

                if (it + 1) % PRINT_EVERY == 0:
                    print(f'\nMoving average norm loss at {it + 1} iterations is: {sum(norm_losses[-10:]) / len(norm_losses[-10:])}. Best norm loss value is: {best_loss_norm}.')
                    print('\nC_PATH mean =', C_PATH.mean(-2))
                    print('\nC_PATH =', C_PATH)

                norm.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)                                
                ptrain_opt.step()

                if it % PTRAIN_LR_DECAY_STEP_SIZE == 0:
                    ptrain_opt.param_groups[0]['lr'] *= PTRAIN_LR_DECAY

            else:
                ELBO_opt.zero_grad()                
                
                # Sample (unknown) theta                
                theta_dict, theta, log_q_theta, parent_loc_scale_dict = q_theta(BATCH_SIZE)
                log_p_theta = priors.log_prob(theta).sum(-1)

                # Fix known theta
                if FIX_THETA_DICT:
                    if platform.python_version() >= '3.9.0':
                        theta_dict = theta_dict | FIX_THETA_DICT
                    else:
                        theta_dict = {**theta_dict, **FIX_THETA_DICT}

                if LEARN_CO2:
                    log_lik, drift, diffusion_sqrt, x_add_CO2 = calc_log_lik2(C_PATH, theta_dict, DT, SBM_SDE, INIT_PRIOR, LEARN_CO2)
                    ELBO = -log_p_theta.mean() + log_q_theta.mean() + log_prob.mean() - log_lik.mean() - obs_model(x_add_CO2, theta_dict)
                else:
                    log_lik, drift, diffusion_sqrt = calc_log_lik2(C_PATH, theta_dict, DT, SBM_SDE, INIT_PRIOR, LEARN_CO2)
                    ELBO = -log_p_theta.mean() + log_q_theta.mean() + log_prob.mean() - log_lik.mean() - obs_model(C_PATH, theta_dict)

                #Negative ELBO: -log p(theta) + log q(theta) - log p(y_0|x_0, theta) [already accounted for in obs_model output when learning x_0] + log q(x|theta) - log p(x|theta) - log p(y|x, theta)
                best_loss_ELBO = ELBO if ELBO < best_loss_ELBO else best_loss_ELBO
                ELBO_losses.append(ELBO.item())

                if (it + 1) % PRINT_EVERY == 0:
                    print(f'\ndrift at {it + 1} iterations: {drift}')
                    print(f'\ndiffusion_sqrt at {it + 1} iterations = {diffusion_sqrt}')
                    print(f'\nMoving average ELBO loss at {it + 1} iterations is: {sum(ELBO_losses[-10:]) / len(ELBO_losses[-10:])}. Best ELBO loss value is: {best_loss_ELBO}.')
                    if LEARN_CO2:
                        print('\nC_PATH with CO2 mean =', x_add_CO2.mean(-2))
                        print('\nC_PATH with CO2 =', x_add_CO2)
                    else:
                        print('\nC_PATH mean =', C_PATH.mean(-2))
                        print('\nC_PATH =', C_PATH)
                    print('\ntheta_dict means: ', {key: theta_dict[key].mean() for key in param_names})
                    print('\nparent_loc_scale_dict: ', parent_loc_scale_dict)

                ELBO.backward()
                torch.nn.utils.clip_grad_norm_(ELBO_params, 5.0)
                ELBO_opt.step()

                if it % ELBO_LR_DECAY_STEP_SIZE == 0:
                    ELBO_opt.param_groups[0]['lr'] *= ELBO_LR_DECAY

            if DEBUG_SAVE_DIR:
                to_save = {'model': net, 'model_state_dict': net.state_dict(), 'ELBO_opt_state_dict': ELBO_opt.state_dict(), 
                        'ptrain_opt_state_dict': ptrain_opt.state_dict(), 'q_theta': q_theta}
                debug_saver.save(to_save, it + 1)

            tq.update()

    print('\nAll finished! Now, we need to check outputs to see if things worked...')            
    
    return net, q_theta, priors, obs_model, norm_losses, ELBO_losses, SBM_SDE

def train_NN_old(DEVICE, NN_ELBO_LR: float, ELBO_ITER: int, BATCH_SIZE: int,
        OBS_CSV_STR: str, OBS_ERROR_SCALE: float, T: float, DT: float, N: int,
        T_SPAN_TENSOR: torch.Tensor, I_S_TENSOR: torch.Tensor, I_D_TENSOR: torch.Tensor, TEMP_TENSOR: torch.Tensor, TEMP_REF: float,
        SBM_SDE_CLASS: str, DIFFUSION_TYPE: str, INIT_PRIOR: torch.distributions.distribution.Distribution,
        PARAMS_DICT: DictOfNpArrays, LEARN_CO2: bool = False, BYPASS_NAN: bool = False,
        NN_ELBO_LR_DECAY: float = 0.8, NN_ELBO_LR_DECAY_STEP_SIZE: int = 50000, PTRAIN_LR_DECAY: float = 0.8, PTRAIN_LR_DECAY_STEP_SIZE: int = 1000,
        PRINT_EVERY: int = 100, DEBUG_SAVE_DIR: str = None, PTRAIN_ITER: int = 0, PTRAIN_LR: float = None, PTRAIN_ALG: BoolAndString = False,
        NUM_LAYERS: int = 5):

    #Sum to get total training iterations.
    T_ITER = ELBO_ITER + PTRAIN_ITER

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
    if LEARN_CO2:
        obs_dim = SBM_SDE.state_dim + 1
    else:
        obs_dim = SBM_SDE.state_dim
    obs_times, obs_means, obs_error = csv_to_obs_df(OBS_CSV_STR, obs_dim, T, OBS_ERROR_SCALE) #csv_to_obs_df function in obs_and_flow module
    obs_model = ObsModel(TIMES = obs_times, DT = DT, MU = obs_means, SCALE = obs_error).to(DEVICE)

    #Establish neural network.
    net = SDEFlow(DEVICE, obs_model, SBM_SDE.state_dim, T, DT, N, num_layers = NUM_LAYERS).to(DEVICE)
    
    #Initiate model debugging saver.
    if DEBUG_SAVE_DIR:
        debug_saver = ModelSaver(save_dir = DEBUG_SAVE_DIR)

    param_names = list(PARAMS_DICT.keys())

    #Record loss throughout training
    best_loss_norm = 1e15
    best_loss_ELBO = 1e15
    norm_losses = []
    ELBO_losses = []

    #Initiate optimizers.
    if PTRAIN_ALG and PTRAIN_ITER != 0:
        ptrain_opt = optim.Adam(net.parameters(), lr = PTRAIN_LR)
        if LEARN_CO2:
            mean_state_obs = torch.mean(obs_model.mu[:-1, :], -1)[None, None, :]
        else:
            mean_state_obs = torch.mean(obs_model.mu, -1)[None, None, :]
    elif not PTRAIN_ALG and PTRAIN_ITER != 0:
        raise Error('Pre-training iterations specified without PTRAIN_ALG input. Must request PTRAIN_ALG = "L1" or "L2".')

    #Separate neural network optimizer.
    nn_ELBO_params = net.parameters()
    nn_ELBO_opt = optim.Adamax(nn_ELBO_params, lr = NN_ELBO_LR)

    #Convert dictionary of fixed parameter Numpy arrays to [1, num_params] tensor and dictionary of size [1] tensors.
    theta = torch.tensor([v.item() for v in list(PARAMS_DICT.values())])[None, :] # torch.Size([1, num_params])
    theta_dict = {k: torch.tensor(v).unsqueeze(0) for k, v in PARAMS_DICT.items()}
    
    #Training loop
    # if BYPASS_NAN:
    #         torch.autograd.set_detect_anomaly(True)
    print(f'\nStarting autoencoder training. {PTRAIN_ITER} pre-training iterations and {ELBO_ITER} ELBO training iterations for {T_ITER} total iterations specified.')    
    net.train()
    with tqdm(total = T_ITER, desc = f'Learning SDE and hidden parameters.', position = -1) as tq:
        for it in range(T_ITER):
            
            C_PATH, log_prob = net(BATCH_SIZE) #Obtain paths with solutions to times including t0.
            
            #NaN handling            
            nan_count = 0
            #Check for NaNs in x.
            if torch.isnan(C_PATH).any():
                if BYPASS_NAN:
                    nan_count += 1
                    print(f'\nnan_count = {nan_count}')
                    print(f'\nWarning. NaN in x at niter: {it}. Using `torch.nan_to_num` to bypass. Check gradient clipping and learning rate to start.')
                    C_PATH = torch.nan_to_num(C_PATH)
                else:
                    raise ValueError(f'\nnan in x at niter: {it}. Check gradient clipping and learning rate to start.')

            if PTRAIN_ALG and it < PTRAIN_ITER:
                ptrain_opt.zero_grad()

                if PTRAIN_ALG == 'L1':
                    l1_norm_element = C_PATH - mean_state_obs #Compute difference between x and observed state means.
                    norm = torch.sum(torch.abs(l1_norm_element), (-1, -2)).mean() #Take L1 mean across all samples.
                    best_loss_norm = norm if norm < best_loss_norm else best_loss_norm
                    norm_losses.append(norm.item())

                elif PTRAIN_ALG == 'L2':
                    l2_norm_element = C_PATH - mean_state_obs #Compute difference between x and observed state means.
                    norm = torch.sqrt(torch.sum(torch.square(l2_norm_element), (-1, -2))).mean() #Take L2 mean across all samples.
                    best_loss_norm = norm if norm < best_loss_norm else best_loss_norm
                    norm_losses.append(norm.item())

                if (it + 1) % PRINT_EVERY == 0:
                    print(f'\nMoving average norm loss at {it + 1} iterations is: {sum(norm_losses[-10:]) / len(norm_losses[-10:])}. Best norm loss value is: {best_loss_norm}.')
                    print('\nC_PATH mean =', C_PATH.mean(-2))
                    print('\nC_PATH =', C_PATH)

                norm.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)                                
                ptrain_opt.step()

                if it % PTRAIN_LR_DECAY_STEP_SIZE == 0:
                    ptrain_opt.param_groups[0]['lr'] *= PTRAIN_LR_DECAY

            else:
                nn_ELBO_opt.zero_grad()

                # Compute likelihood and ELBO
                # Negative ELBO: -log p(theta) + log q(theta) - log p(y_0|x_0, theta) [already accounted for in obs_model output when learning x_0] + log q(x|theta) - log p(x|theta) - log p(y|x, theta)
                if LEARN_CO2:
                    log_lik, drift, diffusion_sqrt, x_add_CO2 = calc_log_lik2(C_PATH, theta_dict, DT, SBM_SDE, INIT_PRIOR, LEARN_CO2)
                    ELBO = log_prob.mean() - log_lik.mean() #- obs_model(x_add_CO2, theta_dict)
                else:
                    log_lik, drift, diffusion_sqrt = calc_log_lik2(C_PATH, theta_dict, DT, SBM_SDE, INIT_PRIOR, LEARN_CO2)
                    ELBO = log_prob.mean() - log_lik.mean() #- obs_model(C_PATH, theta_dict)

                # Record ELBO history and best ELBO so far
                best_loss_ELBO = ELBO if ELBO < best_loss_ELBO else best_loss_ELBO
                ELBO_losses.append(ELBO.item())

                # Print info
                if (it + 1) % PRINT_EVERY == 0:
                    print(f'\ndrift at {it + 1} iterations: {drift}')
                    print(f'\ndiffusion_sqrt at {it + 1} iterations = {diffusion_sqrt}')
                    print(f'\nMoving average ELBO loss at {it + 1} iterations is: {sum(ELBO_losses[-10:]) / len(ELBO_losses[-10:])}. Best ELBO loss value is: {best_loss_ELBO}.')
                    if LEARN_CO2:
                        print('\nC_PATH with CO2 mean =', x_add_CO2.mean(-2))
                        print('\nC_PATH with CO2 =', x_add_CO2)
                    else:
                        print('\nC_PATH mean =', C_PATH.mean(-2))
                        print('\nC_PATH =', C_PATH)

                # Take gradient step
                ELBO.backward()
                torch.nn.utils.clip_grad_norm_(nn_ELBO_params, 5.0)
                nn_ELBO_opt.step()

                if it % NN_ELBO_LR_DECAY_STEP_SIZE == 0:
                    nn_ELBO_opt.param_groups[0]['lr'] *= NN_ELBO_LR_DECAY

            if DEBUG_SAVE_DIR:
                to_save = {'model': net, 'model_state_dict': net.state_dict(), 'ELBO_opt_state_dict': nn_ELBO_opt.state_dict(), 
                        'ptrain_opt_state_dict': ptrain_opt.state_dict()}
                debug_saver.save(to_save, it + 1)

            tq.update()

    print('\nAll finished! Now, we need to check outputs to see if things worked...')
    
    return net, obs_model, norm_losses, ELBO_losses, SBM_SDE
