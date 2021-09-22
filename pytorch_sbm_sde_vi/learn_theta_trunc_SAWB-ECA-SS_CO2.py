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

#SBM temperature forcing parameters
temp_ref = 283
temp_rise = 5 #High estimate of 5 celsius temperature rise by 2100.

#Training parameters
niter = 290000
train_lr = 2e-5 #ELBO learning rate
batch_size = 40 #3 - number needed to fit UCI HPC3 RAM requirements with 16 GB RAM at t = 5000.
eval_batch_size = 40
obs_error_scale = 0.1 #Observation (y) standard deviation.
prior_scale_factor = 0.333 #Proportion of prior standard deviation to prior means.
num_layers = 5 #5 - number needed to fit UCI HPC3 RAM requirements with 16 GB RAM at t = 5000.

#Specify desired SBM SDE model type and details.
state_dim_SAWB_ECA = 4
SBM_SDE_class = 'SAWB-ECA'
diffusion_type = 'SS'
learn_CO2 = True
theta_dist = 'TruncatedNormal' #String needs to be exact name of the distribution class. Options are 'TruncatedNormal' and 'RescaledLogitNormal'.

#Parameter prior means
u_Q_ref_mean = 0.2
Q_mean = 0.001
a_MSA_mean = 0.5
K_DE_mean = 1850
K_UE_mean = 0.2
V_DE_ref_mean = 0.16
V_UE_ref_mean = 0.012
Ea_V_DE_mean = 65
Ea_V_UE_mean = 55
r_M_mean = 0.0018
r_E_mean = 0.00003
r_L_mean = 0.000008
s_SOC_mean = 0.005
s_DOC_mean = 0.005
s_MBC_mean = 0.005
s_EEC_mean = 0.005

#SAWB-ECA theta truncated normal distribution parameter details in order of mean, sdev, lower, and upper.
u_Q_ref_details = torch.Tensor([u_Q_ref_mean, u_Q_ref_mean * prior_scale_factor, 0, 1])
Q_details = torch.Tensor([Q_mean, Q_mean * prior_scale_factor, 0, 1])
a_MSA_details = torch.Tensor([a_MSA_mean, a_MSA_mean * prior_scale_factor, 0, 1])
K_DE_details = torch.Tensor([K_DE_mean, K_DE_mean * prior_scale_factor, 0, 10000])
K_UE_details = torch.Tensor([K_UE_mean, K_UE_mean * prior_scale_factor, 0, 100])
V_DE_ref_details = torch.Tensor([V_DE_ref_mean, V_DE_ref_mean * prior_scale_factor, 0, 10])
V_UE_ref_details = torch.Tensor([V_UE_ref_mean, V_UE_ref_mean * prior_scale_factor, 0, 1])
Ea_V_DE_details = torch.Tensor([Ea_V_DE_mean, Ea_V_DE_mean * prior_scale_factor, 10, 150])
Ea_V_UE_details = torch.Tensor([Ea_V_UE_mean, Ea_V_UE_mean * prior_scale_factor, 10, 150])
r_M_details = torch.Tensor([r_M_mean, r_M_mean * prior_scale_factor, 0, 1])
r_E_details = torch.Tensor([r_E_mean, r_M_mean * prior_scale_factor, 0, 1])
r_L_details = torch.Tensor([r_L_mean, r_M_mean * prior_scale_factor, 0, 1])

#SAWB-ECA-SS diffusion matrix parameter distribution details
s_SOC_details = torch.Tensor([s_SOC_mean, s_SOC_mean * prior_scale_factor, 0, 1])
s_DOC_details = torch.Tensor([s_DOC_mean, s_DOC_mean * prior_scale_factor, 0, 1])
s_MBC_details = torch.Tensor([s_MBC_mean, s_MBC_mean * prior_scale_factor, 0, 1])
s_EEC_details = torch.Tensor([s_EEC_mean, s_EEC_mean * prior_scale_factor, 0, 1])

SAWB_ECA_SS_priors_details = {'u_Q_ref': u_Q_ref_details, 'Q': Q_details, 'a_MSA': a_MSA_details, 'K_DE': K_DE_details, 'K_UE': K_UE_details, 'V_DE_ref': V_DE_ref_details, 'V_UE_ref': V_UE_ref_details, 'Ea_V_DE': Ea_V_DE_details, 'Ea_V_UE': Ea_V_UE_details, 'r_M': r_M_details, 'r_E': r_E_details, 'r_L': r_L_details, 's_SOC': s_SOC_details, 's_DOC': s_DOC_details, 's_MBC': s_MBC_details, 's_EEC': s_EEC_details}

#Initial condition prior means
x0_SAWB_ECA = [65, 0.4, 2.5, 0.3]
x0_SAWB_ECA_tensor = torch.tensor(x0_SAWB_ECA).to(active_device)
x0_prior_SAWB_ECA = D.multivariate_normal.MultivariateNormal(x0_SAWB_ECA_tensor, scale_tril = torch.eye(state_dim_SAWB_ECA).to(active_device) * obs_error_scale * x0_SAWB_ECA_tensor)

#Generate exogenous input vectors.
#Obtain temperature forcing function.
temp_tensor = temp_gen(t_span_tensor, temp_ref, temp_rise).to(active_device)

#Obtain SOC and DOC pool litter input vectors for use in flow SDE functions.
i_s_tensor = i_s(t_span_tensor).to(active_device) #Exogenous SOC input function
i_d_tensor = i_d(t_span_tensor).to(active_device) #Exogenous DOC input function

#Generate observation model.
csv_data_path = os.path.join('generated_data/', 'SAWB-ECA-SS_CO2_trunc_sample_y_t_1000_dt_0-01_sd_scale_0-333.csv')

#Call training loop function for SCON-C.
net, q_theta, p_theta, obs_model, ELBO_hist, list_parent_loc_scale, SBM_SDE_instance = train2(
        active_device, train_lr, niter, batch_size, num_layers,
        csv_data_path, obs_error_scale, t, dt_flow, n, 
        t_span_tensor, i_s_tensor, i_d_tensor, temp_tensor, temp_ref,
        SBM_SDE_class, diffusion_type, x0_prior_SAWB_ECA, SAWB_ECA_SS_priors_details, learn_CO2,
        theta_dist, LR_DECAY = 0.9, DECAY_STEP_SIZE = 25000, PRINT_EVERY = 50)

#Save net and ELBO files.
now = datetime.now()
now_string = 'SAWB-ECA-SS_CO2_trunc' + now.strftime('_%Y_%m_%d_%H_%M_%S')
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
true_theta = torch.load('generated_data/SAWB-ECA-SS_CO2_trunc_sample_y_t_1000_dt_0-01_sd_scale_0-333_rsample.pt', map_location = active_device)

#Plot training posterior results and ELBO history.
net.eval()
x, _ = net(eval_batch_size)
plots_folder = 'training_plots/'
plot_elbo(ELBO_hist, niter, t, dt_flow, batch_size, eval_batch_size, num_layers, train_lr, prior_scale_factor, plots_folder, now_string, xmin = int(niter * 0.2))
plot_states_post(x, q_theta, obs_model, SBM_SDE_instance, niter, t, dt_flow, batch_size, eval_batch_size, num_layers, train_lr, prior_scale_factor, plots_folder, now_string, learn_CO2, ymin_list = [0, 0, 0, 0, 0], ymax_list = [100., 8, 10., 4., 0.15])
plot_theta(p_theta, q_theta, true_theta, niter, t, dt_flow, batch_size, eval_batch_size, num_layers, train_lr, prior_scale_factor, plots_folder, now_string)
