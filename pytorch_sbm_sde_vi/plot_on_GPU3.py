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

#PyTorch settings
torch.manual_seed(0)
print('cuda device available?: ', torch.cuda.is_available())
active_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.set_printoptions(precision = 8)

#IAF SSM time parameters
dt_flow = 1.0 #Increased from 0.1 to reduce memory.
t = 5000 #In hours.

#Training parameters
niter = 230000
train_lr = 2e-5 #ELBO learning rate
batch_size = 32 #32 is presently max batch_size with 16 GB VRAM at t = 5000.
eval_batch_size = 32
obs_error_scale = 0.1 #Observation (y) standard deviation.
prior_scale_factor = 0.333 #Proportion of prior standard deviation to prior means.
num_layers = 5 #5 - number needed to fit UCI HPC3 RAM requirements with 16 GB RAM at t = 5000.
learn_CO2 = True

now_string = 'SCON-SS_CO2_trunc_5000_diff_theta_2021_10_15_12_53_43'
outputs_folder = 'training_pt_outputs/'
plots_folder = 'training_plots/'
save_string = '_iter_230000_t_5000_dt_1.0_batch_32_layers_5_lr_2e-05_sd_scale_0.333_SCON-SS_CO2_trunc_5000_diff_theta_2021_10_15_12_53_43.pt'

net_save_string = os.path.join(outputs_folder, 'net' + save_string)
q_theta_save_string = os.path.join(outputs_folder, 'q_theta' + save_string)
p_theta_save_string = os.path.join(outputs_folder, 'p_theta' + save_string)
obs_model_save_string = os.path.join(outputs_folder, 'obs_model' + save_string)
SBM_SDE_instance_save_string = os.path.join(outputs_folder, 'SBM_SDE_instance' + save_string)

#Load .pt files.
net = torch.load(net_save_string)
net.to(active_device)
p_theta = torch.load(p_theta_save_string)
q_theta = torch.load(q_theta_save_string)
q_theta.to(active_device)
obs_model = torch.load(obs_model_save_string)
obs_model.to(active_device)
SBM_SDE_instance = torch.load(SBM_SDE_instance_save_string)
true_theta = torch.load('generated_data/SCON-SS_CO2_trunc_5000_diff_theta_2021_10_12_16_48_sample_y_t_5000_dt_0-01_sd_scale_0-333_rsample.pt', map_location = active_device)

#Plot training posterior results and ELBO history.
net.eval()
x, _ = net(eval_batch_size)

plot_states_post(x, q_theta, obs_model, SBM_SDE_instance, niter, t, dt_flow, batch_size, eval_batch_size, num_layers, train_lr, prior_scale_factor, plots_folder, now_string, FIX_THETA_DICT = None, LEARN_CO2 = learn_CO2, ymin_list = [0, 0, 0, 0], ymax_list = [150., 20., 40., 0.025])
plot_theta(p_theta, q_theta, true_theta, niter, t, dt_flow, batch_size, eval_batch_size, num_layers, train_lr, prior_scale_factor, plots_folder, now_string)
