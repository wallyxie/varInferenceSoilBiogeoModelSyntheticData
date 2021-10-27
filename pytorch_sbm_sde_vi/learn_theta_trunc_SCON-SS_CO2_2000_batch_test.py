#Python-related imports
import math
import sys
from datetime import datetime
import os.path

#Torch-related imports
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
from SBM_SDE_classes import *
from obs_and_flow import *
from training import *
from plotting import *
from mean_field import *
from TruncatedNormal import *
from LogitNormal import *

#Other imports
from tqdm import tqdm

#PyTorch settings
torch.manual_seed(0)
print('cuda device available?: ', torch.cuda.is_available())
active_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.set_printoptions(precision = 8)

#IAF SSM time parameters
dt_flow = 1.0 #Increased from 0.1 to reduce memory.
t = 2000 #In hours.
n = int(t / dt_flow) + 1
t_span = np.linspace(0, t, n)
t_span_tensor = torch.reshape(torch.Tensor(t_span), [1, n, 1]).to(active_device) #T_span needs to be converted to tensor object. Additionally, facilitates conversion of I_S and I_D to tensor objects.

#SBM temperature forcing parameters
temp_ref = 283
temp_rise = 5 #High estimate of 5 celsius temperature rise by 2100.

#Training parameters
niter = 5
train_lr = 2e-5 #ELBO learning rate
obs_error_scale = 0.1 #Observation (y) standard deviation.
prior_scale_factor = 0.333 #Proportion of prior standard deviation to prior means.
num_layers = 5 #5 - number needed to fit UCI HPC3 RAM requirements with 16 GB RAM at t = 5000.

#Specify desired SBM SDE model type and details.
state_dim_SCON = 3
SBM_SDE_class = 'SCON'
diffusion_type = 'SS'
learn_CO2 = True
theta_dist = 'TruncatedNormal' #String needs to be exact name of the distribution class. Options are 'TruncatedNormal' and 'RescaledLogitNormal'.
fix_dict = None

SCON_SS_priors_details = {k: v.to(active_device) for k, v in torch.load('generated_data/SCON-SS_CO2_trunc_2000_2021_10_06_16_32_sample_y_t_2000_dt_0-01_sd_scale_0-333_hyperparams.pt').items()}

#Initial condition prior means
x0_SCON = [65, 0.4, 2.5]
x0_SCON_tensor = torch.tensor(x0_SCON).to(active_device)
x0_prior_SCON = D.multivariate_normal.MultivariateNormal(x0_SCON_tensor, scale_tril = torch.eye(state_dim_SCON).to(active_device) * obs_error_scale * x0_SCON_tensor)

#Generate exogenous input vectors.
#Obtain temperature forcing function.
temp_tensor = temp_gen(t_span_tensor, temp_ref, temp_rise).to(active_device)

#Obtain SOC and DOC pool litter input vectors for use in flow SDE functions.
i_s_tensor = i_s(t_span_tensor).to(active_device) #Exogenous SOC input function
i_d_tensor = i_d(t_span_tensor).to(active_device) #Exogenous DOC input function

#Generate observation model.
csv_data_path = os.path.join('generated_data/', 'SCON-SS_CO2_trunc_2000_2021_10_06_16_32_sample_y_t_2000_dt_0-01_sd_scale_0-333.csv')

#Call training loop function for SCON-SS.
for batch_size in range(40, 100): 
    print('Trying batch_size = ', batch_size)
    net, q_theta, p_theta, obs_model, ELBO_hist, list_parent_loc_scale, SBM_SDE_instance = train2(
        active_device, train_lr, niter, batch_size, num_layers,
        csv_data_path, obs_error_scale, t, dt_flow, n, 
        t_span_tensor, i_s_tensor, i_d_tensor, temp_tensor, temp_ref,
        SBM_SDE_class, diffusion_type, x0_prior_SCON, SCON_SS_priors_details, fix_dict, learn_CO2,
        theta_dist, BYPASS_NAN = False, LR_DECAY = 0.9, DECAY_STEP_SIZE = 25000, PRINT_EVERY = 10)
