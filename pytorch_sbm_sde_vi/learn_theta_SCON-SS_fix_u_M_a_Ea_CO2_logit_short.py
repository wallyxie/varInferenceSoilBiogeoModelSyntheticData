#Python-related imports
import math
import sys
from datetime import datetime
import os.path

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
from SBM_SDE_classes import *
from obs_and_flow import *
from training import *
from plotting import *
from mean_field import *
#from TruncatedNormal import *
from LogitNormal import *

#Other imports
from tqdm import tqdm

#PyTorch settings
active_device = torch.device('cpu')
if torch.cuda.is_available():
    print('CUDA device detected.')
    active_device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print('No CUDA device detected.')
    raise EnvironmentError

torch.set_printoptions(precision = 8)
torch.manual_seed(0)

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
niter = 410000
ptrain_iter = 0
train_lr = 1.5e-5 #ELBO learning rate
ptrain_lr = 1e-5
batch_size = 31
eval_batch_size = 31
obs_error_scale = 0.1 #Observation (y) standard deviation.
prior_scale_factor = 0.25 #Proportion of prior standard deviation to prior means.
num_layers = 5 #5 - number needed to fit UCI HPC3 RAM requirements with 16 GB RAM at t = 5000.

#Specify desired SBM SDE model type and details.
state_dim_SCON = 3
SBM_SDE_class = 'SCON'
diffusion_type = 'SS'
learn_CO2 = True
theta_dist = 'RescaledLogitNormal' #String needs to be exact name of the distribution class. Options are 'TruncatedNormal' and 'RescaledLogitNormal'.
fix_dict = {k: v.to(active_device) for k, v in torch.load(os.path.join('generated_data/', 'SCON-SS_CO2_logit_short_fix_u_M_a_Ea_2021_11_21_14_46_sample_y_t_5000_dt_0-01_sd_scale_0-25_fix_dict.pt')).items()}

#Load parameterization of priors.
SCON_SS_priors_details = {k: v.to(active_device) for k, v in torch.load(os.path.join('generated_data/', 'SCON-SS_CO2_logit_short_fix_u_M_a_Ea_2021_11_21_14_46_sample_y_t_5000_dt_0-01_sd_scale_0-25_hyperparams.pt')).items()}

#Initial condition prior means
x0_SCON_tensor = torch.load(os.path.join('generated_data/', 'SCON-SS_CO2_logit_short_fix_u_M_a_Ea_2021_11_21_14_46_sample_y_t_5000_dt_0-01_sd_scale_0-25_x0_SCON_tensor.pt')).to(active_device)
x0_prior_SCON = D.multivariate_normal.MultivariateNormal(x0_SCON_tensor, scale_tril = torch.eye(state_dim_SCON).to(active_device) * obs_error_scale * x0_SCON_tensor)

#Generate exogenous input vectors.
#Obtain temperature forcing function.
temp_tensor = temp_gen(t_span_tensor, temp_ref, temp_rise).to(active_device)

#Obtain SOC and DOC pool litter input vectors for use in flow SDE functions.
i_s_tensor = i_s(t_span_tensor).to(active_device) #Exogenous SOC input function
i_d_tensor = i_d(t_span_tensor).to(active_device) #Exogenous DOC input function

#Generate observation model.
csv_data_path = os.path.join('generated_data/', 'SCON-SS_CO2_logit_short_fix_u_M_a_Ea_2021_11_21_14_46_sample_y_t_5000_dt_0-01_sd_scale_0-25.csv')

#Call training loop function for SCON-SS.
net, q_theta, p_theta, obs_model, norm_hist, ELBO_hist, SBM_SDE_instance = train2(
        active_device, train_lr, niter, batch_size, num_layers,
        csv_data_path, obs_error_scale, t, dt_flow, n, 
        t_span_tensor, i_s_tensor, i_d_tensor, temp_tensor, temp_ref,
        SBM_SDE_class, diffusion_type, x0_prior_SCON, SCON_SS_priors_details, fix_dict, learn_CO2,
        THETA_DIST = theta_dist, BYPASS_NAN = False, LR_DECAY = 0.95, DECAY_STEP_SIZE = 25000, PRINT_EVERY = 1,
        DEBUG_SAVE_DIR = None, PTRAIN_ITER = ptrain_iter, PTRAIN_LR = ptrain_lr, PTRAIN_ALG = 'L2')
print('Training finished. Moving to saving of output files.')

#Save net and ELBO files.
now = datetime.now()
now_string = 'SCON-SS_fix_u_M_a_Ea_CO2_logit_short' + now.strftime('_%Y_%m_%d_%H_%M_%S')
save_string = f'_iter_{niter}_piter_{ptrain_iter}_t_{t}_dt_{dt_flow}_batch_{batch_size}_layers_{num_layers}_lr_{train_lr}_sd_scale_{prior_scale_factor}_{now_string}.pt'
outputs_folder = 'training_pt_outputs/'
net_save_string = os.path.join(outputs_folder, 'net' + save_string)
net_state_dict_save_string = os.path.join(outputs_folder,'net_state_dict' + save_string)
q_theta_save_string = os.path.join(outputs_folder, 'q_theta' + save_string)
p_theta_save_string = os.path.join(outputs_folder, 'p_theta' + save_string)
obs_model_save_string = os.path.join(outputs_folder, 'obs_model' + save_string)
norm_save_string = os.path.join(outputs_folder, 'norm' + save_string)
ELBO_save_string = os.path.join(outputs_folder, 'ELBO' + save_string)
SBM_SDE_instance_save_string = os.path.join(outputs_folder, 'SBM_SDE_instance' + save_string)
torch.save(net, net_save_string)
torch.save(net.state_dict(), net_state_dict_save_string) #For loading net on CPU.
torch.save(q_theta, q_theta_save_string)
torch.save(p_theta, p_theta_save_string)
torch.save(obs_model, obs_model_save_string)
torch.save(norm_hist, norm_save_string)
torch.save(ELBO_hist, ELBO_save_string)
torch.save(SBM_SDE_instance, SBM_SDE_instance_save_string)
print('Output files saving finished. Moving to plotting.')

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
true_theta = torch.load(os.path.join('generated_data/', 'SCON-SS_CO2_logit_short_fix_u_M_a_Ea_2021_11_21_14_46_sample_y_t_5000_dt_0-01_sd_scale_0-25_rsample.pt'), map_location = active_device)

#Plot training posterior results and ELBO history.
net.eval()
x, _ = net(eval_batch_size)
plots_folder = 'training_plots/'
plot_elbo(ELBO_hist, niter, ptrain_iter, t, dt_flow, batch_size, eval_batch_size, num_layers, train_lr, prior_scale_factor, plots_folder, now_string, xmin = int(niter * 0.2))
print('ELBO plotting finished.')
plot_states_post(x, q_theta, obs_model, SBM_SDE_instance, niter, ptrain_iter, t, dt_flow, batch_size, eval_batch_size, num_layers, train_lr, prior_scale_factor, plots_folder, now_string, learn_CO2, ymin_list = [0, 0, 0, 0], ymax_list = [50., 7., 10., 0.025])
print('States fit plotting finished.')
plot_theta(p_theta, q_theta, true_theta, niter, ptrain_iter, t, dt_flow, batch_size, eval_batch_size, num_layers, train_lr, prior_scale_factor, plots_folder, now_string)
print('Prior-posterior pair plotting finished.')
