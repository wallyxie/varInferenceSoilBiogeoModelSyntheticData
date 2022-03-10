#Python-related imports
import math
import sys
from datetime import datetime
import os.path
import time

#Torch imports
import torch
import torch.distributions as D
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function

#PyData imports
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

#Module imports
from training import *
from plotting import *

#PyTorch settings
if torch.cuda.is_available():
    print('CUDA device detected.')
    active_device = torch.device('cuda')
else:
    print('No CUDA device detected.')
    raise EnvironmentError

torch.set_printoptions(precision = 8)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

#IAF SSM time parameters
dt_flow = 1.0 #Increased from 0.1 to reduce memory.
t = 5000 #In hours.
n = int(t / dt_flow) + 1
t_span = np.linspace(0, t, n)
t_span_tensor = torch.reshape(torch.Tensor(t_span), [1, n, 1]).to(active_device) #T_span needs to be converted to tensor object. Additionally, facilitates conversion of I_S and I_D to tensor objects.

#SBM temperature forcing parameters
temp_ref = 283
temp_rise = 5 #High estimate of 5 celsius temperature rise by 2100.

#Training parameters
elbo_iter = 10
elbo_lr = 5e-3
elbo_lr_decay = 0.7
elbo_lr_decay_step_size = 10000
elbo_warmup_iter = 5
elbo_warmup_lr = 1e-6
ptrain_iter = 0
ptrain_alg = 'L1'
obs_error_scale = 0.1
prior_scale_factor = 0.25
num_layers = 5
reverse = True
base_state = False

#Specify desired SBM SDE model type and details.
state_dim_SCON = 3
SBM_SDE_class = 'SCON'
diffusion_type = 'SS'
learn_CO2 = True
theta_dist = 'RescaledLogitNormal' #String needs to be exact name of the distribution class. Options are 'TruncatedNormal' and 'RescaledLogitNormal'.
fix_theta_dict = None

#Load parameterization of priors.
SCON_SS_priors_details = {k: v.to(active_device) for k, v in torch.load(os.path.join('generated_data/', 'SCON-SS_CO2_logit_short_2021_11_17_20_16_sample_y_t_5000_dt_0-01_sd_scale_0-25_hyperparams.pt')).items()}

#Initial condition prior means
x0_SCON_tensor = torch.load(os.path.join('generated_data/', 'SCON-SS_CO2_logit_short_2021_11_17_20_16_sample_y_t_5000_dt_0-01_sd_scale_0-25_x0_SCON_tensor.pt')).to(active_device)
x0_prior_SCON = D.multivariate_normal.MultivariateNormal(x0_SCON_tensor, scale_tril = torch.eye(state_dim_SCON).to(active_device) * obs_error_scale * x0_SCON_tensor)

#Generate exogenous input vectors.
#Obtain temperature forcing function.
temp_tensor = temp_gen(t_span_tensor, temp_ref, temp_rise).to(active_device)

#Obtain SOC and DOC pool litter input vectors for use in flow SDE functions.
i_s_tensor = i_s(t_span_tensor).to(active_device) #Exogenous SOC input function
i_d_tensor = i_d(t_span_tensor).to(active_device) #Exogenous DOC input function

#Assign path to observations .csv file.
csv_data_path = os.path.join('generated_data/', 'SCON-SS_CO2_logit_short_2021_11_17_20_16_sample_y_t_5000_dt_0-01_sd_scale_0-25.csv')

#Call training loop function.
t_total = 0
for batch_size in range(20, 50):
    torch.cuda.empty_cache()
    print('Trying batch_size = ', batch_size)
    t_start = time.process_time()
    net, q_theta, p_theta, obs_model, norm_hist, ELBO_hist, SBM_SDE_instance = train(
        active_device, elbo_lr, elbo_iter, batch_size,
        csv_data_path, obs_error_scale, t, dt_flow, n, 
        t_span_tensor, i_s_tensor, i_d_tensor, temp_tensor, temp_ref,
        SBM_SDE_class, diffusion_type, x0_prior_SCON,
        SCON_SS_priors_details, fix_theta_dict, learn_CO2, theta_dist, 
        ELBO_WARMUP_ITER = elbo_warmup_iter, ELBO_WARMUP_INIT_LR = elbo_warmup_lr, ELBO_LR_DECAY = elbo_lr_decay, ELBO_LR_DECAY_STEP_SIZE = elbo_lr_decay_step_size,
        PRINT_EVERY = 10, VERBOSE = True,
        DEBUG_SAVE_DIR = None, PTRAIN_ITER = ptrain_iter, PTRAIN_ALG = ptrain_alg,
        NUM_LAYERS = num_layers, REVERSE = reverse, BASE_STATE = base_state)
    t_end = time.process_time()
    t_total += t_end - t_start
    print(f'Total function time by batch_size = {batch_size} is t_total = {t_total} seconds')
