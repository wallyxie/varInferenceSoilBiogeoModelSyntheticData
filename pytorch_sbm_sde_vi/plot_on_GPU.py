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

#Module module imports
from SBM_SDE_classes import *
from obs_and_flow import *
from training import *
from plotting import *
from mean_field import *
from TruncatedNormal import *
from LogitNormal import *

#IAF SSM time parameters
dt_flow = 1.0 #Increased from 0.1 to reduce memory.
t = 1000 #In hours.
n = int(t / dt_flow) + 1
t_span = np.linspace(0, t, n)
t_span_tensor = torch.reshape(torch.Tensor(t_span), [1, n, 1]).to(active_device) #T_span needs to be converted to tensor object. Additionally, facilitates conversion of I_S and I_D to tensor objects.

true_theta = torch.load('generated_data/SCON-SS_CO2_logit_alt_2021_09_27_18_54_sample_y_t_1000_dt_0-01_sd_scale_0-333_rsample.pt', map_location = active_device)

now_string = 'SCON-SS_CO2_trunc_2021_09_28_11_32_36'
outputs_folder = 'training_pt_outputs/'
save_string = '250000_t_1000_dt_1.0_batch_45_layers_5_lr_2e-05_sd_scale_0.333_SCON-SS_CO2_trunc_2021_09_28_11_32_36'

net_save_string = os.path.join(outputs_folder, 'net' + save_string)
q_theta_save_string = os.path.join(outputs_folder, 'q_theta' + save_string)
p_theta_save_string = os.path.join(outputs_folder, 'p_theta' + save_string)
obs_model_save_string = os.path.join(outputs_folder, 'obs_model' + save_string)
ELBO_save_string = os.path.join(outputs_folder, 'ELBO' + save_string)
SBM_SDE_instance_save_string = os.path.join(outputs_folder, 'SBM_SDE_instance' + save_string)

#Plot training posterior results and ELBO history.
net.eval()
x, _ = net(eval_batch_size)

plot_states_post(x, q_theta, obs_model, SBM_SDE_instance, niter, t, dt_flow, batch_size, eval_batch_size, num_layers, train_lr, prior_scale_factor, plots_folder, now_string, learn_CO2, ymin_list = [0, 0, 0, 0], ymax_list = [100., 15., 15., 0.1])
plot_theta(p_theta, q_theta, true_theta, niter, t, dt_flow, batch_size, eval_batch_size, num_layers, train_lr, prior_scale_factor, plots_folder, now_string)
