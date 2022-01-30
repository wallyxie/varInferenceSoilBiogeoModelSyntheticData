#Python-related imports
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
from training_minibatch import *
from plotting import * #Need to update versions of plotting scripts for minibatching.

#PyTorch settings
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
dt_flow = 1.0
t = 5000 #In hours.
n = int(t / dt_flow) + 1 #Total number of time-steps needs to include x0.
t_span = np.linspace(0, t, n)
t_span_tensor = torch.reshape(torch.Tensor(t_span), [1, n, 1]).to(active_device) #t_span needs to be converted to tensor object. Additionally, facilitates conversion of I_S and I_D to tensor objects.

#SBM temperature forcing parameters
temp_ref = 283
temp_rise = 5 #High estimate of 5 celsius temperature rise by 2100.

#Training parameters
elbo_iter = 410000
elbo_lr = 2e-5 #ELBO learning rate
elbo_lr_decay = 0.9
elbo_decay_step_size = 25000
ptrain_iter = 0
ptrain_lr = 1e-4
ptrain_alg = 'L1'
ptrain_lr_decay = 0.9
ptrain_decay_step_size = 1000
batch_size = 31
eval_batch_size = 31
obs_error_scale = 0.1
prior_scale_factor = 0.25
num_layers = 5
kernel_size = 3
num_resblocks = 2
minibatch_t = 5000
theta_cond = 'convolution'
other_cond_inputs = True

#Specify desired SBM SDE model type and details.
state_dim_SCON = 3
SBM_SDE_class = 'SCON'
diffusion_type = 'SS'
learn_CO2 = True
theta_dist = 'RescaledLogitNormal' #String needs to be exact name of the distribution class. Options are 'TruncatedNormal' and 'RescaledLogitNormal'.
fix_theta_dict = None

#Load parameterization of priors.
SCON_SS_priors_details = {k: v.to(active_device) for k, v in torch.load(os.path.join('generated_data/', 'SCON-SS_CO2_logit_short_2021_11_17_20_16_sample_y_t_5000_dt_0-01_sd_scale_0-25_hyperparams.pt')).items()}

#Initial condition prior means
x0_SCON_tensor = torch.load(os.path.join('generated_data/', 'SCON-SS_CO2_logit_short_2021_11_17_20_16_sample_y_t_5000_dt_0-01_sd_scale_0-25_x0_SCON_tensor.pt')).to(active_device)
x0_prior_SCON = D.multivariate_normal.MultivariateNormal(x0_SCON_tensor, scale_tril = torch.eye(state_dim_SCON).to(active_device) * obs_error_scale * x0_SCON_tensor)

#Generate exogenous input vectors.
#Obtain temperature forcing function.
temp_tensor = temp_gen(t_span_tensor, temp_ref, temp_rise).to(active_device)

#Obtain SOC and DOC pool litter input vectors for use in flow SDE functions.
i_s_tensor = i_s(t_span_tensor).to(active_device) #Exogenous SOC input function
i_d_tensor = i_d(t_span_tensor).to(active_device) #Exogenous DOC input function

#Assign path to observations .csv file.
csv_data_path = os.path.join('generated_data/', 'SCON-SS_CO2_logit_short_2021_11_17_20_16_sample_y_t_5000_dt_0-01_sd_scale_0-25.csv')

#Call training loop function for SCON-SS.
net, q_theta, p_theta, obs_model, norm, ELBO, SBM_SDE_instance = train_minibatch(
        active_device, elbo_lr, elbo_iter, batch_size,
        csv_data_path, obs_error_scale, t, dt_flow, n,
        t_span_tensor, i_s_tensor, i_d_tensor, temp_tensor, temp_ref,
        SBM_SDE_class, diffusion_type, x0_prior_SCON,
        SCON_SS_priors_details, fix_theta_dict, learn_CO2, theta_dist,
        ELBO_LR_DECAY = elbo_lr_decay, ELBO_LR_DECAY_STEP_SIZE = elbo_decay_step_size, PTRAIN_LR_DECAY = ptrain_lr_decay, PTRAIN_LR_DECAY_STEP_SIZE = ptrain_decay_step_size,
        PRINT_EVERY = 1, DEBUG_SAVE_DIR = None, PTRAIN_ITER = ptrain_iter, PTRAIN_LR = ptrain_lr, PTRAIN_ALG = ptrain_alg,
        MINIBATCH_T = minibatch_t, NUM_LAYERS = num_layers, KERNEL_SIZE = kernel_size, NUM_RESBLOCKS = num_resblocks,
        THETA_COND = theta_cond, OTHER_COND_INPUTS = other_cond_inputs)
print('Training finished. Moving to saving of output files.')

#Save net and ELBO files.
now = datetime.now()
now_string = 'SCON-SS_CO2_minibatch_logit_short' + now.strftime('_%Y_%m_%d_%H_%M_%S')
save_string = f'_iter_{elbo_iter}_piter_{ptrain_iter}_t_{t}_dt_{dt_flow}_minibatch_{minibatch_t}_batch_{batch_size}_layers_{num_layers}_lr_{elbo_lr}_sd_scale_{prior_scale_factor}_{now_string}.pt'
outputs_folder = 'training_pt_outputs/'
net_save_string = os.path.join(outputs_folder, 'net' + save_string)
net_state_dict_save_string = os.path.join(outputs_folder,'net_state_dict' + save_string)
q_theta_save_string = os.path.join(outputs_folder, 'q_theta' + save_string)
p_theta_save_string = os.path.join(outputs_folder, 'p_theta' + save_string)
obs_model_save_string = os.path.join(outputs_folder, 'obs_model' + save_string)
norm_save_string = os.path.join(outputs_folder, 'norm' + save_string)
ELBO_save_string = os.path.join(outputs_folder, 'ELBO' + save_string)
SBM_SDE_instance_save_string = os.path.join(outputs_folder, 'SBM_SDE_instance' + save_string)
torch.save(train_args, train_args_save_string, _use_new_zipfile_serialization = False)
torch.save(net, net_save_string, _use_new_zipfile_serialization = False)
torch.save(net.state_dict(), net_state_dict_save_string, _use_new_zipfile_serialization = False) #For loading net on CPU.
torch.save(q_theta, q_theta_save_string, _use_new_zipfile_serialization = False)
torch.save(p_theta, p_theta_save_string, _use_new_zipfile_serialization = False)
torch.save(obs_model, obs_model_save_string, _use_new_zipfile_serialization = False)
torch.save(ELBO_hist, ELBO_save_string, _use_new_zipfile_serialization = False)
torch.save(SBM_SDE_instance, SBM_SDE_instance_save_string, _use_new_zipfile_serialization = False)
print('Output files saving finished. Moving to plotting.')

#Release some CUDA memory and load .pt files.
torch.cuda.empty_cache()
#net = torch.load(net_save_string)
#net.to(active_device)
p_theta = torch.load(p_theta_save_string)
q_theta = torch.load(q_theta_save_string)
q_theta.to(active_device)
#obs_model = torch.load(obs_model_save_string)
#obs_model.to(active_device)
#ELBO = torch.load(ELBO_save_string)
#SBM_SDE_instance = torch.load(SBM_SDE_instance_save_string)
true_theta = torch.load(os.path.join('generated_data/', 'SCON-SS_CO2_logit_short_2021_11_17_20_16_sample_y_t_5000_dt_0-01_sd_scale_0-25_rsample.pt'), map_location = active_device)

#Plot training posterior results and ELBO history.
#net.eval()
#x, _ = net(eval_batch_size)
plots_folder = 'training_plots/'
#plot_elbo(ELBO, elbo_iter, ptrain_iter, t, dt_flow, batch_size, eval_batch_size, num_layers, elbo_lr, prior_scale_factor, plots_folder, now_string, xmin = int(elbo_iter * 0.2))
#print('ELBO plotting finished.')
#plot_states_post(x, q_theta, obs_model, SBM_SDE_instance, elbo_iter, ptrain_iter, t, dt_flow, batch_size, eval_batch_size, num_layers, elbo_lr, prior_scale_factor, plots_folder, now_string, learn_CO2, ymin_list = [0, 0, 0, 0], ymax_list = [60., 5., 8., 0.025])
#print('States fit plotting finished.')
plot_theta(p_theta, q_theta, true_theta, elbo_iter, elbo_warmup_iter, t, dt_flow, batch_size, eval_batch_size, num_layers, elbo_lr, elbo_warmup_lr, prior_scale_factor, plots_folder, now_string)
print('Prior-posterior pair plotting finished.')
