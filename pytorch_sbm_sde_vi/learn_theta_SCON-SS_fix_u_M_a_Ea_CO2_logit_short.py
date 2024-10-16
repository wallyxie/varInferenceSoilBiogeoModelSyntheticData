#Python-related imports
import math
import sys
from datetime import datetime
import os.path
import time

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
from training_log_p import *
from plotting import *

#PyTorch settings
if torch.cuda.is_available():
    print('CUDA device detected.')
    active_device = torch.device('cuda')
else:
    print('No CUDA device detected.')
    raise EnvironmentError

torch.set_printoptions(precision = 8)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

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
elbo_iter = 160000
elbo_lr = 5e-3
elbo_lr_decay = 0.75
elbo_lr_decay_step_size = 5000
elbo_warmup_iter = 10000
elbo_warmup_lr = 1e-6
ptrain_iter = 0
ptrain_alg = 'L1'
batch_size = 31
eval_batch_size = 250
obs_error_scale = 0.1
prior_scale_factor = 0.25
num_layers = 5
reverse = True
base_state = False

train_args = {'t': t, 'dt_flow': dt_flow, 'elbo_iter': elbo_iter, 'elbo_lr': elbo_lr, 'elbo_lr_decay': elbo_lr_decay, 'elbo_lr_decay_step_size': elbo_lr_decay_step_size, 'elbo_warmup_iter': elbo_warmup_iter, 'elbo_warmup_lr': elbo_warmup_lr, 'ptrain_iter': ptrain_iter, 'ptrain_alg': ptrain_alg, 'batch_size': batch_size, 'obs_error_scale': obs_error_scale, 'prior_scale_factor': prior_scale_factor, 'num_layers': num_layers, 'reverse': reverse, 'base_state': base_state}

#Specify desired SBM SDE model type and details.
state_dim_SCON = 3
SBM_SDE_class = 'SCON'
diffusion_type = 'SS'
learn_CO2 = True
theta_dist = 'RescaledLogitNormal' #String needs to be exact name of the distribution class. Options are 'TruncatedNormal' and 'RescaledLogitNormal'.
fix_theta_dict = {k: v.to(active_device) for k, v in torch.load(os.path.join('generated_data/', 'SCON-SS_CO2_logit_short_fix_u_M_a_Ea_2021_11_21_14_46_sample_y_t_5000_dt_0-01_sd_scale_0-25_fix_dict.pt')).items()}

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

#Assign path to observations .csv file.
csv_data_path = os.path.join('generated_data/', 'SCON-SS_CO2_logit_short_fix_u_M_a_Ea_2021_11_21_14_46_sample_y_t_5000_dt_0-01_sd_scale_0-25.csv')

start_time = time.process_time()
#Call training loop function.
net, q_theta, p_theta, obs_model, norm_hist, ELBO_hist, times_per_iter_hist, SBM_SDE_instance, best_train_ELBO = train(
        active_device, elbo_lr, elbo_iter, batch_size,
        csv_data_path, obs_error_scale, t, dt_flow, n,
        t_span_tensor, i_s_tensor, i_d_tensor, temp_tensor, temp_ref,
        SBM_SDE_class, diffusion_type, x0_prior_SCON,
        SCON_SS_priors_details, fix_theta_dict, learn_CO2, theta_dist,
        ELBO_WARMUP_ITER = elbo_warmup_iter, ELBO_WARMUP_INIT_LR = elbo_warmup_lr, ELBO_LR_DECAY = elbo_lr_decay, ELBO_LR_DECAY_STEP_SIZE = elbo_lr_decay_step_size,
        PRINT_EVERY = 20, VERBOSE = True,
        DEBUG_SAVE_DIR = None, PTRAIN_ITER = ptrain_iter, PTRAIN_ALG = ptrain_alg,
        NUM_LAYERS = num_layers, REVERSE = reverse, BASE_STATE = base_state)
elapsed_time = time.process_time() - start_time
print(f'Training finished after {elapsed_time} seconds. Moving to saving of output files.')

#Save net and ELBO files.
now = datetime.now()
now_string = 'SCON-SS_fix_u_M_a_Ea_CO2_logit_short' + now.strftime('_%Y_%m_%d_%H_%M_%S')
save_string = f'_iter_{elbo_iter}_warmup_{elbo_warmup_iter}_t_{t}_dt_{dt_flow}_batch_{batch_size}_layers_{num_layers}_lr_{elbo_lr}_decay_step_{elbo_lr_decay_step_size}_warmup_lr_{elbo_warmup_lr}_sd_scale_{prior_scale_factor}_{now_string}.pt'
outputs_folder = 'training_pt_outputs/'
train_args_save_string = os.path.join(outputs_folder, 'train_args' + save_string)
net_save_string = os.path.join(outputs_folder, 'net' + save_string)
net_state_dict_save_string = os.path.join(outputs_folder,'net_state_dict' + save_string)
q_theta_save_string = os.path.join(outputs_folder, 'q_theta' + save_string)
q_theta_state_dict_save_string = os.path.join(outputs_folder, 'q_theta_state_dict' + save_string)
p_theta_save_string = os.path.join(outputs_folder, 'p_theta' + save_string)
obs_model_save_string = os.path.join(outputs_folder, 'obs_model' + save_string)
ELBO_save_string = os.path.join(outputs_folder, 'ELBO' + save_string)
times_per_iter_save_string = os.path.join(outputs_folder, 'times_per_iter' + save_string)
SBM_SDE_instance_save_string = os.path.join(outputs_folder, 'SBM_SDE_instance' + save_string)
best_train_ELBO_save_string = os.path.join(outputs_folder, 'best_train_ELBO' + f'_iter_{elbo_iter}_warmup_{elbo_warmup_iter}_t_{t}_dt_{dt_flow}_batch_{batch_size}_layers_{num_layers}_lr_{elbo_lr}_decay_step_{elbo_lr_decay_step_size}_warmup_lr_{elbo_warmup_lr}_sd_scale_{prior_scale_factor}_{now_string}.txt')
elapsed_time_save_string = os.path.join(outputs_folder, 'elapsed_time' + f'_iter_{elbo_iter}_warmup_{elbo_warmup_iter}_t_{t}_dt_{dt_flow}_batch_{batch_size}_layers_{num_layers}_lr_{elbo_lr}_decay_step_{elbo_lr_decay_step_size}_warmup_lr_{elbo_warmup_lr}_sd_scale_{prior_scale_factor}_{now_string}.txt')
torch.save(train_args, train_args_save_string)
torch.save(net, net_save_string)
torch.save(net.state_dict(), net_state_dict_save_string) #For loading net on CPU.
torch.save(q_theta, q_theta_save_string)
torch.save(q_theta.state_dict(), q_theta_state_dict_save_string)
torch.save(p_theta, p_theta_save_string)
torch.save(obs_model, obs_model_save_string)
torch.save(ELBO_hist, ELBO_save_string)
torch.save(times_per_iter_hist, times_per_iter_save_string)
torch.save(SBM_SDE_instance, SBM_SDE_instance_save_string)
with open(best_train_ELBO_save_string, 'w') as f:
    print(f'Best train ELBO: {best_train_ELBO}', file = f)
with open(elapsed_time_save_string, 'w') as f:
    print(f'Elapsed time: {elapsed_time} seconds', file = f)

#Compute test ELBO and log p.
net.eval()
with torch.no_grad():
    x, log_prob = net(eval_batch_size)
    print('x = ', x)
    theta_dict, theta, log_q_theta, parent_loc_scale_dict = q_theta(eval_batch_size)
    log_p_theta = p_theta.log_prob(theta).sum(-1)
    if fix_theta_dict:
        if platform.python_version() >= '3.9.0':
            theta_dict = theta_dict | fix_theta_dict
        else:
            theta_dict = {**theta_dict, **fix_theta_dict}
    if learn_CO2:
        log_lik, drift, diffusion_sqrt, x_add_CO2 = calc_log_lik(x, theta_dict, dt_flow, SBM_SDE_instance, x0_prior_SCON, learn_CO2)
        neg_ELBO = -log_p_theta.mean() + log_q_theta.mean() + log_prob.mean() - log_lik.mean() - obs_model(x_add_CO2)
        log_p = -log_p_theta.mean() - log_lik.mean() - obs_model(x_add_CO2)
    else:
        log_lik, drift, diffusion_sqrt = calc_log_lik(x, theta_dict, dt_flow, SBM_SDE_instance, x0_prior_SCON, learn_CO2)
        neg_ELBO = -log_p_theta.mean() + log_q_theta.mean() + log_prob.mean() - log_lik.mean() - obs_model(x)
        log_p = -log_p_theta.mean() - log_lik.mean() - obs_model(x)
    print('x.size() =', x.size())
    print(f'Net with {train_args} has test neg_ELBO = {neg_ELBO} and log p = {log_p}')
    test_elbo_and_log_p_save_string = os.path.join(outputs_folder, 'test_elbo_and_log_p' + f'_iter_{elbo_iter}_warmup_{elbo_warmup_iter}_t_{t}_dt_{dt_flow}_batch_{batch_size}_layers_{num_layers}_lr_{elbo_lr}_decay_step_{elbo_lr_decay_step_size}_warmup_lr_{elbo_warmup_lr}_sd_scale_{prior_scale_factor}_{now_string}.txt')
    with open(test_elbo_and_log_p_save_string, 'w') as f:
        print(f'Test ELBO: {neg_ELBO}\nlog p: {log_p}', file = f)

#Save net.eval() samples from trained net object for CPU plotting and processing.
batch_multiples = 1
eval_batch_size_save = eval_batch_size #testing batch size for saved x samples
with torch.no_grad():
    for i in range(batch_multiples):
        print(i)
        _x, _ = net(eval_batch_size_save)
        _x.detach().cpu()
        if learn_CO2:
            q_theta_sample_dict, _, _, _ = q_theta(_x.size(0))
            if fix_theta_dict:
                q_theta_sample_dict = {**q_theta_sample_dict, **fix_theta_dict}
            _x = SBM_SDE_instance.add_CO2(_x, q_theta_sample_dict) #Add CO2 to x tensor if CO2 is being fit.
        if i == 0:
            x_eval = _x
        else:
            x_eval = torch.cat([x_eval, _x], 0)
        del _x
        torch.cuda.empty_cache()
        print(torch.cuda.memory_allocated())
        print(torch.cuda.memory_reserved())
        print(x_eval.size())
print(x_eval)
x_eval_save_string = os.path.join(outputs_folder, 'x_eval' + save_string)
torch.save(x_eval, x_eval_save_string)

print('Output files saving finished. Moving to plotting.')
#Plot training posterior results and ELBO history.
with torch.no_grad():
    x, _ = net(eval_batch_size)
plots_folder = 'training_plots/'
plot_elbo(ELBO_hist, elbo_iter, elbo_warmup_iter, t, dt_flow, batch_size, eval_batch_size, num_layers, elbo_lr, elbo_lr_decay_step_size, elbo_warmup_lr, prior_scale_factor, plots_folder, now_string, xmin = elbo_warmup_iter + int(elbo_iter / 4))
print('ELBO plotting finished.')
plot_states_post(x, q_theta, obs_model, SBM_SDE_instance, elbo_iter, elbo_warmup_iter, t, dt_flow, batch_size, eval_batch_size, num_layers, elbo_lr, elbo_lr_decay_step_size, elbo_warmup_lr, prior_scale_factor, plots_folder, now_string, fix_theta_dict, learn_CO2, ymin_list = [0, 0, 0, 0], ymax_list = [55., 8., 10., 0.03])
print('States fit plotting finished.')
true_theta = torch.load(os.path.join('generated_data/', 'SCON-SS_CO2_logit_short_fix_u_M_a_Ea_2021_11_21_14_46_sample_y_t_5000_dt_0-01_sd_scale_0-25_rsample.pt'), map_location = active_device)
plot_theta(p_theta, q_theta, true_theta, elbo_iter, elbo_warmup_iter, t, dt_flow, batch_size, eval_batch_size, num_layers, elbo_lr, elbo_lr_decay_step_size, elbo_warmup_lr, prior_scale_factor, plots_folder, now_string)
print('Prior-posterior pair plotting finished.')
