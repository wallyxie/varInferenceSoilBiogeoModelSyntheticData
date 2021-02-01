import torch
import torch.nn as nn
import torch.distributions as d
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#SBM-related scripts
from sbm_temp_functions import *
from sbm_steady_state_init_functions import *
from sbm_sde import *

torch.manual_seed(0)
STATE_DIM = 3
dt = .1
T = 1000 #Run simulation for 1,000 hours.
N = int(T / dt)
T_span = np.linspace(0, T, N)
T_span_tensor = torch.Tensor(T_span)[(None,) * 2] #T_span needs to be converted to tensor object.

BATCH_SIZE = 1
device = torch.device("".join(["cuda:",f'{args.CUDA_ID}']) if torch.cuda.is_available() else "cpu")
LR = 1e-3
niter = 1000000

#Read in data
obs_df = pd.read_csv('CON_synthetic_sol_df.csv')
obs_times = torch.Tensor(np.array(obs_df['hour'])) #Extract data observation times.
obs_means = torch.Tensor(np.array(obs_df.drop(columns = 'hour'))) #Convert C data to tensor.

def calc_negative_elbo_scon(log_prob, x, x0, T_span, litter_drift_and_diffusion_scon, drift_and_diffusion_scon, scon_params_dict):
    #T_range = torch.Tensor(T_range)[(None,) * 2] #T_range needs to be converted to tensor object.
    x[:, :, 0] = x0 #Set initial conditions
    litter_drift, litter_diffusion = litter_drift_and_diffusion_scon(x[:, :, :-1], T_span)
    system_drift, system_diffusion = drift_and_diffusion_scon(x[:, :, :-1], T_span, scon_params_dict, temp_ref)
    #Euler-Maruyama modified from basic form because of exogenous input with separate noise.
    #euler_maruyama = d.multivariate_normal.MultivariateNormal(loc = x[:, :, :-1].permute(0, 2, 1) + system_drift * dt, scale_tril = system_diffusion * math.sqrt(dt)) + d.multivariate_normal.MultivariateNormal(loc = x[:, :, :-1].permute(0, 2, 1) + litter_drift * dt, scale_tril = litter_diffusion * math.sqrt(dt)) 
    euler_maruyama = d.multivariate_normal.MultivariateNormal(loc = x[:, :, :-1].permute(0, 2, 1) + system_drift * dt, scale_tril = system_diffusion * math.sqrt(dt)) #Need to add time-dependent exogenous input.
    return (log_prob - torch.sum(euler_maruyama.log_prob(x[:, :, 1:].permute(0, 2, 1)).sum(-1), -1, keepdim=True)).mean()
