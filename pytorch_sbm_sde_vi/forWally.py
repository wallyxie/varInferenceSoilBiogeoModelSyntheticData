import torch
from torch.autograd import Function
from torch import nn
import torch.distributions as D
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os

from obs_and_flow import SDEFlowWP, ObsModel, csv_to_obs_df

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
dt_flow = 1. 
t = 1000 
n = int(t / dt_flow) + 1
t_span = np.linspace(0, t, n)
t_span_tensor = torch.reshape(torch.Tensor(t_span), [1, n, 1]).to(device) 

obs_error_scale = 0.1
num_layers = 5
state_dims = 3

csv_data_path = os.path.join('generated_data/', 'SCON-C_no_CO2_logit_alt_sample_y_t_1000_dt_0-01_sd_scale_0-333.csv')
obs_times, obs_means, obs_error = csv_to_obs_df(csv_data_path, state_dims, t, obs_error_scale)

q_theta = D.normal.Normal(loc=torch.zeros(3, device=device), scale=torch.ones(3, device=device))

obs_model = ObsModel(device, obs_times, dt_flow, obs_means, SCALE=1.0)
net = SDEFlowWP(device, obs_model, q_theta, state_dims, t, dt_flow, n, num_layers = num_layers).to(device)

x, xlp = net(100)

print(x.shape, xlp)
