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

now_string = 'SCON-C_CO2_logit_short_2022_02_24_15_27_58'
outputs_folder = 'training_pt_outputs/'
plots_folder = 'training_plots/'
save_string = '_iter_120000_warmup_5000_t_5000_dt_1.0_batch_31_layers_5_lr_0.01_decay_step_5000_warmup_lr_1e-06_sd_scale_0.25_SCON-C_CO2_logit_short_2022_02_24_15_27_58.pt'

obs_model_save_string = os.path.join(outputs_folder, 'obs_model' + save_string)
net_save_string = os.path.join(outputs_folder, 'net' + save_string)
net_state_dict_save_string = os.path.join(outputs_folder, 'net_state_dict' + save_string)
SBM_SDE_instance_save_string = os.path.join(outputs_folder, 'SBM_SDE_instance' + save_string)
q_theta_save_string = os.path.join(outputs_folder, 'q_theta' + save_string)
train_args_save_string = os.path.join(outputs_folder, 'train_args' + save_string)

#Training parameters
train_args = torch.load(train_args_save_string, map_location = active_device)
t = train_args['t']
dt_flow = train_args['dt_flow']
n = int(t / dt_flow) + 1
t_span = np.linspace(0, t, n)
t_span_tensor = torch.reshape(torch.Tensor(t_span), [1, n, 1]).to(active_device) #T_span needs to be converted to tensor object. Additionally, facilitates conversion of I_S and I_D to tensor objects.
elbo_iter = train_args['elbo_iter']
elbo_lr = train_args['elbo_lr']
elbo_lr_decay = train_args['elbo_lr_decay']
elbo_lr_decay_step_size = train_args['elbo_lr_decay_step_size']
elbo_warmup_iter = train_args['elbo_warmup_iter']
elbo_warmup_lr = train_args['elbo_warmup_lr']
ptrain_iter = train_args['ptrain_iter']
ptrain_alg = train_args['ptrain_alg']
batch_size = train_args['batch_size']
eval_batch_size = 31
obs_error_scale = train_args['obs_error_scale']
prior_scale_factor = train_args['prior_scale_factor']
num_layers = train_args['num_layers']
reverse = train_args['reverse']
base_state = train_args['base_state']

#Specify desired SBM SDE model type and details.
state_dim_SCON = 3
learn_CO2 = True
theta_dist = 'RescaledLogitNormal' #String needs to be exact name of the distribution class. Options are 'TruncatedNormal' and 'RescaledLogitNormal'.
fix_theta_dict = None

#Load .pt files.
obs_model = torch.load(obs_model_save_string, map_location = active_device)
net = SDEFlow(active_device, obs_model, state_dim_SCON, t, dt_flow, n, NUM_LAYERS = num_layers, REVERSE = reverse, BASE_STATE = base_state)
net.load_state_dict(torch.load(net_state_dict_save_string, map_location = active_device))
q_theta = torch.load(q_theta_save_string, map_location = active_device)
SBM_SDE = torch.load(SBM_SDE_instance_save_string, map_location = active_device)

#Save evaluation samples from trained net object.
net.eval()
batch_multiples = 10
with torch.no_grad():
    for i in range(batch_multiples):
        print(i)
        _x, _ = net(eval_batch_size)
        _x.detach().cpu()
        if learn_CO2:
            q_theta_sample_dict, _, _, _ = q_theta(_x.size(0))
            if fix_theta_dict:
                q_theta_sample_dict = {**q_theta_sample_dict, **FIX_THETA_DICT}
            _x = SBM_SDE.add_CO2(_x, q_theta_sample_dict) #Add CO2 to x tensor if CO2 is being fit.
        if i == 0:
            x = _x
        else:
            x = torch.cat([x, _x], 0)
        del _x
        torch.cuda.empty_cache()
        print(torch.cuda.memory_allocated())
        print(torch.cuda.memory_reserved())
        print(x.size())
print(x)
x_save_string = os.path.join(outputs_folder, 'x' + save_string)
torch.save(x, x_save_string)
