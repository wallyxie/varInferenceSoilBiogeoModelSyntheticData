#Python-related imports
import math, sys
from datetime import datetime
import os.path

#Torch-related imports
import torch
import torch.distributions as D
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from TruncatedNormal import *
from LogitNormal import *

#PyData imports
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

#Model-specific module imports
from SBM_SDE_tensor import *
from obs_and_flow import *
from training import *
from plotting import *
from mean_field import *

#Other imports
from tqdm import tqdm

#PyTorch settings
torch.manual_seed(0)
print('cuda device available?: ', torch.cuda.is_available())
active_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.set_printoptions(precision = 8)

#Neural SDE parameters
dt_flow = 1.0 #Increased from 0.1 to reduce memory.
t = 1000 #In hours.
n = int(t / dt_flow) + 1
t_span = np.linspace(0, t, n)
t_span_tensor = torch.reshape(torch.Tensor(t_span), [1, n, 1]).to(active_device) #T_span needs to be converted to tensor object. Additionally, facilitates conversion of I_S and I_D to tensor objects.
state_dim_SCON = 3 #Not including CO2 in STATE_DIM, because CO2 is an observation.
state_dim_SAWB = 4 #Not including CO2 in STATE_DIM, because CO2 is an observation.

#SBM temperature forcing parameters
temp_ref = 283
temp_rise = 5 #High estimate of 5 celsius temperature rise by 2100.

#Training parameters
niter = 405000
piter = 0
pretrain_lr = 1e-3 #Norm regularization learning rate
train_lr = 5e-4 #ELBO learning rate
batch_size = 32 #3 - number needed to fit UCI HPC3 RAM requirements with 16 GB RAM at t = 5000.
eval_batch_size = 32
obs_error_scale = 0.1 #Observation (y) standard deviation.
prior_scale_factor = 0.333 #Proportion of prior standard deviation to prior means.
num_layers = 5 #5 - number needed to fit UCI HPC3 RAM requirements with 16 GB RAM at t = 5000.
theta_dist = 'TruncatedNormal' #String needs to be exact name of the distribution class. Other option is 'RescaledLogitNormal'.

#SCON theta truncated normal prior distribution parameter details in order of mean, lower, and upper. Distribution sdev assumed to be some proportion of the mean. 
k_S_ref_details = torch.Tensor([0.0005, 0.0005 * prior_scale_factor, 0, 0.01])
k_D_ref_details = torch.Tensor([0.0008, 0.0008 * prior_scale_factor, 0, 0.01])
k_M_ref_details = torch.Tensor([0.0007, 0.0007 * prior_scale_factor, 0, 0.01])

#SCON-C diffusion matrix parameter truncated normal prior distribution parameter details in order of mean, lower, and upper. 
c_SOC_details = torch.Tensor([0.1, 0.1 * prior_scale_factor, 0, 1]).to(active_device)
c_DOC_details = torch.Tensor([0.002, 0.002 * prior_scale_factor, 0, 0.02]).to(active_device)
c_MBC_details = torch.Tensor([0.002, 0.002 * prior_scale_factor, 0, 0.02]).to(active_device)

SCONR_C_fix_u_M_a_Ea_priors_details = {'k_S_ref': k_S_ref_details, 'k_D_ref': k_D_ref_details, 'k_M_ref': k_M_ref_details, 'c_SOC': c_SOC_details, 'c_DOC': c_DOC_details, 'c_MBC': c_MBC_details}

#Initial condition prior means
x0_SCON = [65, 0.4, 2.5]
x0_SCON_tensor = torch.tensor(x0_SCON).to(active_device)
x0_prior_SCON = D.multivariate_normal.MultivariateNormal(x0_SCON_tensor,
                                                         scale_tril = torch.eye(state_dim_SCON).to(active_device) * obs_error_scale * x0_SCON_tensor)

#Generate exogenous input vectors.
#Obtain temperature forcing function.
temp_tensor = temp_gen(t_span_tensor, temp_ref, temp_rise).to(active_device)

#Obtain SOC and DOC pool litter input vectors for use in flow SDE functions.
i_s_tensor = i_s(t_span_tensor).to(active_device) #Exogenous SOC input function
i_d_tensor = i_d(t_span_tensor).to(active_device) #Exogenous DOC input function

#Generate observation model.
csv_data_path = os.path.join('generated_data/', 'SCONR_C_fix_u_M_a_Ea_trunc_sample_y_t_1000_dt_0-01_sd_scale_0-333.csv')
obs_times, obs_means_noCO2, obs_error = csv_to_obs_df(csv_data_path, state_dim_SCON, t, obs_error_scale)
obs_model = ObsModel(active_device, TIMES = obs_times, DT = dt_flow, MU = obs_means_noCO2, SCALE = obs_error).to(active_device) 

#Call training loop function for SCON-C.
net, q_theta, p_theta, obs_model, ELBO_hist, list_parent_loc_scale = train(
        active_device, pretrain_lr, train_lr, niter, piter, batch_size, num_layers,
        state_dim_SCON, csv_data_path, obs_error_scale, t, dt_flow, n, 
        t_span_tensor, i_s_tensor, i_d_tensor, temp_tensor, temp_ref,
        drift_diffusion_SCONR_C_fix_a_Ea, x0_prior_SCON, SCONR_C_fix_u_M_a_Ea_priors_details, theta_dist,
        LEARN_THETA = True, LR_DECAY = 0.85, DECAY_STEP_SIZE = 50000, PRINT_EVERY = 100)

#Save net and ELBO files.
now = datetime.now()
now_string = 'SCONR_C_fix_u_M_a_Ea_trunc_' + now.strftime('%Y_%m_%d_%H_%M_%S')
save_string = f'_iter_{niter}_t_{t}_dt_{dt_flow}_batch_{batch_size}_layers_{num_layers}_lr_{train_lr}_sd_scale_{prior_scale_factor}_{now_string}.pt'
outputs_folder = 'training_pt_outputs/'
net_save_string = os.path.join(outputs_folder, 'net' + save_string)
net_state_dict_save_string = os.path.join(outputs_folder,'net_state_dict' + save_string)
q_theta_save_string = os.path.join(outputs_folder, 'q_theta' + save_string)
p_theta_save_string = os.path.join(outputs_folder, 'p_theta' + save_string)
obs_model_save_string = os.path.join(outputs_folder, 'obs_model' + save_string)
ELBO_save_string = os.path.join(outputs_folder, 'ELBO' + save_string)
list_parent_loc_scale_save_string = os.path.join(outputs_folder, 'parent_loc_scale_trajectory' + save_string)
torch.save(net, net_save_string)
torch.save(net.state_dict(), net_state_dict_save_string) #For loading net on CPU.
torch.save(q_theta, q_theta_save_string)
torch.save(p_theta, p_theta_save_string)
torch.save(obs_model, obs_model_save_string) 
torch.save(ELBO_hist, ELBO_save_string)
torch.save(list_parent_loc_scale, list_parent_loc_scale_save_string)

#Release some CUDA memory and load .pt files.
torch.cuda.empty_cache()
net = torch.load(net_save_string)
net.to(active_device)
obs_model = torch.load(obs_model_save_string)
obs_model.to(active_device)
ELBO_hist = torch.load(ELBO_save_string)

#Plot training posterior results and ELBO history.
net.eval()
x, _ = net(eval_batch_size)
plots_folder = 'training_plots/'
plot_elbo(ELBO_hist, niter, piter, t, dt_flow, batch_size, eval_batch_size, num_layers, train_lr, prior_scale_factor, plots_folder, now_string, xmin = int((niter - piter) * 0.2)) #xmin < (niter - piter).
plot_states_post(x, obs_model, state_dim_SCON, niter, piter, t, dt_flow, batch_size, eval_batch_size, num_layers, train_lr, prior_scale_factor, plots_folder, now_string, ymin_list = [0, 0, 0], ymax_list = [100., 12., 12.])
plot_theta(p_theta, q_theta, niter, piter, t, dt_flow, batch_size, eval_batch_size, num_layers, train_lr, prior_scale_factor, plots_folder, now_string)
