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
t = 5000 #In hours.
n = int(t / dt_flow) + 1
t_span = np.linspace(0, t, n)
t_span_tensor = torch.reshape(torch.Tensor(t_span), [1, n, 1]).to(active_device) #T_span needs to be converted to tensor object. Additionally, facilitates conversion of I_S and I_D to tensor objects.

#SBM temperature forcing parameters
temp_ref = 283
temp_rise = 5 #High estimate of 5 celsius temperature rise by 2100.

#Training parameters
niter = 250000
train_lr = 2e-5 #ELBO learning rate
batch_size = 32 #32 is presently max batch_size with 16 GB VRAM at t = 5000 so far.
eval_batch_size = 32
obs_error_scale = 0.1 #Observation (y) standard deviation.
prior_scale_factor = 0.333 #Proportion of prior standard deviation to prior means.
num_layers = 5

#Specify desired SBM SDE model type and details.
state_dim_SCON = 3
SBM_SDE_class = 'SCON'
diffusion_type = 'SS'
learn_CO2 = True
theta_dist = 'TruncatedNormal' #String needs to be exact name of the distribution class. Options are 'TruncatedNormal' and 'RescaledLogitNormal'.

SCON_SS_priors_details = {k: v.to(active_device) for k, v in torch.load('generated_data/SCON-SS_fix_u_M_a_Ea_CO2_trunc_5000_diff_theta_2021_10_19_17_24_sample_y_t_5000_dt_0-01_sd_scale_0-333_hyperparams.pt').items()}

SCON_SS_fix_u_M_a_Ea_dict = {k: v.to(active_device) for k, v in torch.load('generated_data/SCON-SS_fix_u_M_a_Ea_CO2_trunc_5000_diff_theta_2021_10_19_17_24_sample_y_t_5000_dt_0-01_sd_scale_0-333_fix_dict.pt').items()}

#Initial condition prior means
x0_SCON = [60, 18, 8]
x0_SCON_tensor = torch.tensor(x0_SCON).to(active_device)
x0_prior_SCON = D.multivariate_normal.MultivariateNormal(x0_SCON_tensor, scale_tril = torch.eye(state_dim_SCON).to(active_device) * obs_error_scale * x0_SCON_tensor)

#Generate exogenous input vectors.
#Obtain temperature forcing function.
temp_tensor = temp_gen(t_span_tensor, temp_ref, temp_rise).to(active_device)

#Obtain SOC and DOC pool litter input vectors for use in flow SDE functions.
i_s_tensor = i_s(t_span_tensor).to(active_device) #Exogenous SOC input function
i_d_tensor = i_d(t_span_tensor).to(active_device) #Exogenous DOC input function

#Generate observation model.
csv_data_path = os.path.join('generated_data/', 'SCON-SS_fix_u_M_a_Ea_CO2_trunc_5000_diff_theta_2021_10_19_17_24_sample_y_t_5000_dt_0-01_sd_scale_0-333.csv')

#Call training loop function for SCON-SS.
net, q_theta, p_theta, obs_model, ELBO_hist, list_parent_loc_scale, SBM_SDE_instance = train2(
        active_device, train_lr, niter, batch_size, num_layers,
        csv_data_path, obs_error_scale, t, dt_flow, n, 
        t_span_tensor, i_s_tensor, i_d_tensor, temp_tensor, temp_ref,
        SBM_SDE_class, diffusion_type, x0_prior_SCON, SCON_SS_priors_details, SCON_SS_fix_u_M_a_Ea_dict, learn_CO2,
        theta_dist, BYPASS_NAN = False, LR_DECAY = 0.92, DECAY_STEP_SIZE = 25000, PRINT_EVERY = 50)

#Save net and ELBO files.
now = datetime.now()
now_string = 'SCON-SS_fix_u_M_a_Ea_CO2_trunc' + now.strftime('_%Y_%m_%d_%H_%M_%S')
save_string = f'_iter_{niter}_t_{t}_dt_{dt_flow}_batch_{batch_size}_layers_{num_layers}_lr_{train_lr}_sd_scale_{prior_scale_factor}_{now_string}.pt'
outputs_folder = 'training_pt_outputs/'
net_save_string = os.path.join(outputs_folder, 'net' + save_string)
net_state_dict_save_string = os.path.join(outputs_folder,'net_state_dict' + save_string)
q_theta_save_string = os.path.join(outputs_folder, 'q_theta' + save_string)
p_theta_save_string = os.path.join(outputs_folder, 'p_theta' + save_string)
obs_model_save_string = os.path.join(outputs_folder, 'obs_model' + save_string)
ELBO_save_string = os.path.join(outputs_folder, 'ELBO' + save_string)
list_parent_loc_scale_save_string = os.path.join(outputs_folder, 'parent_loc_scale_trajectory' + save_string)
SBM_SDE_instance_save_string = os.path.join(outputs_folder, 'SBM_SDE_instance' + save_string)
torch.save(net, net_save_string)
torch.save(net.state_dict(), net_state_dict_save_string) #For loading net on CPU.
torch.save(q_theta, q_theta_save_string)
torch.save(p_theta, p_theta_save_string)
torch.save(obs_model, obs_model_save_string) 
torch.save(ELBO_hist, ELBO_save_string)
torch.save(list_parent_loc_scale, list_parent_loc_scale_save_string)
torch.save(SBM_SDE_instance, SBM_SDE_instance_save_string)

#Release some CUDA memory and load .pt files.
torch.cuda.empty_cache()
net = torch.load(net_save_string)
net.to(active_device)
p_theta = torch.load(p_theta_save_string)
q_theta = torch.load(q_theta_save_string)
q_theta.to(active_device)
obs_model = torch.load(obs_model_save_string)
obs_model.to(active_device)
ELBO_hist = torch.load(ELBO_save_string)
SBM_SDE_instance = torch.load(SBM_SDE_instance_save_string)
true_theta = torch.load('generated_data/SCON-SS_fix_u_M_a_Ea_CO2_trunc_5000_diff_theta_2021_10_19_17_24_sample_y_t_5000_dt_0-01_sd_scale_0-333_rsample.pt', map_location = active_device)

#Plot training posterior results and ELBO history.
net.eval()
x, _ = net(eval_batch_size)
plots_folder = 'training_plots/'
plot_elbo(ELBO_hist, niter, t, dt_flow, batch_size, eval_batch_size, num_layers, train_lr, prior_scale_factor, plots_folder, now_string, xmin = int(niter * 0.1))
plot_states_post(x, q_theta, obs_model, SBM_SDE_instance, niter, t, dt_flow, batch_size, eval_batch_size, num_layers, train_lr, prior_scale_factor, plots_folder, now_string, SCON_SS_fix_u_M_a_Ea_dict, learn_CO2, ymin_list = [0, 0, 0, 0], ymax_list = [100., 25., 15., 0.04])
plot_theta(p_theta, q_theta, true_theta, niter, t, dt_flow, batch_size, eval_batch_size, num_layers, train_lr, prior_scale_factor, plots_folder, now_string)
