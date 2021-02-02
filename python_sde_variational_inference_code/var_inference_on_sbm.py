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

I_S_tensor = 0.001 + 0.0005 * torch.sin((2 * math.pi / (24 * 365)) * T_span_tensor) #Exogenous SOC input function
I_D_tensor =  0.0001 + 0.00005 * torch.sin((2 * math.pi / (24 * 365)) * T_span_tensor) #Exogenous DOC input function

def calc_negative_elbo_scon(log_prob, T_span, dt, I_S_tensor, I_D_tensor, analytical_steady_state_init_con, drift_and_diffusion_scon, scon_params_dict, path_count):
    T_span_tensor = torch.Tensor(T_span)[(None,) * 2] #T_range needs to be converted to tensor object.
    drift_vector = torch.zeros(path_count, 3, N)
    drift_vector, diffusion_matrix = drift_and_diffusion_scon(drift_vector, T_span_tensor, dt, I_S_tensor, I_D_tensor, analytical_steady_state_init_con, scon_params_dict, temp_ref, path_count)
    print('\n system drift', drift_vector)
    print('\n system diffusion', diffusion_matrix)
    euler_maruyama = d.multivariate_normal.MultivariateNormal(loc = drift_vector.permute(0, 2, 1), scale_tril = diffusion_matrix * math.sqrt(dt))
    return (log_prob - torch.sum(euler_maruyama.log_prob(drift_vector[:, :, :].permute(0, 2, 1)).sum(-1), -1, keepdim=True)).mean()
