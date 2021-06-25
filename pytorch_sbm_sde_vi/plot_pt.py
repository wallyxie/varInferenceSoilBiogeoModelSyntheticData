import math
from tqdm import tqdm
from datetime import datetime

#Torch-related imports
import torch
import torch.distributions as D
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

#Model-specific imports
from SBM_SDE_tensor import *
from obs_and_flow import *
from training import *
from plotting import *
from mean_field import *

#PyTorch settings
torch.manual_seed(0)
print('cuda device available?: ', torch.cuda.is_available())
active_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

#Neural SDE parameters
dt_flow = 0.1 #Increased from 0.1 to reduce memory.
t = 1400 #5000. Reduced to see impact on memory. #In hours.
n = int(t / dt_flow) + 1
t_span = np.linspace(0, t, n)
t_span_tensor = torch.reshape(torch.Tensor(t_span), [1, n, 1]).to(active_device) #T_span needs to be converted to tensor object. Additionally, facilitates conversion of I_S and I_D to tensor objects.
state_dim_SCON = 3 #Not including CO2 in STATE_DIM, because CO2 is an observation.
state_dim_SAWB = 4 #Not including CO2 in STATE_DIM, because CO2 is an observation.

#SBM temperature forcing parameters
temp_ref = 283
temp_rise = 5 #High estimate of 5 celsius temperature rise by 2100.

#Training parameters
niter = 17000
piter = 3500
pretrain_lr = 1e-4 #Norm regularization learning rate
train_lr = 1e-4 #ELBO learning rate
batch_size = 1 #3 - number needed to fit UCI HPC3 RAM requirements with 16 GB RAM at t = 5000.
eval_batch_size = 1
obs_error_scale = 0.1 #Observation (y) standard deviation.
prior_scale_factor = 0.1 #Proportion of prior standard deviation to prior means.
num_layers = 5 #5 - number needed to fit UCI HPC3 RAM requirements with 16 GB RAM at t = 5000.

def plot_elbo(elbo_hist, niter, t, dt, batch_size, eval_batch_size, num_layers, now_string, xmin = 0, ymax = None, yscale = 'linear'):
    iters = torch.arange(xmin + 1, len(elbo_hist) + 1).cpu().detach().numpy()
    plt.plot(iters, elbo_hist[xmin:])
    plt.ylim((None, ymax))
    plt.yscale(yscale)
    plt.ylabel('ELBO')
    plt.xlabel('Iteration')
    plt.title(f'ELBO history after {xmin} iterations')
    plt.savefig(f'ELBO_iter_{niter}_t_{t}_dt_{dt}_batch_{batch_size}_samples_{eval_batch_size}_layers_{num_layers}_{now_string}.png', dpi = 300)
    
def plot_states_post(x, obs_model, niter, t, dt, batch_size, eval_batch_size, num_layers, now_string, ymin_list = None, ymax_list = None, state_dim = 3):
    state_list = ['SOC', 'DOC', 'MBC', 'EEC']   
    fig, axs = plt.subplots(state_dim)

    obs_model.mu = obs_model.mu.cpu().detach().numpy()
    obs_model.scale = obs_model.scale.cpu().detach().numpy()

    for i in range(state_dim):
        q_mean, q_std = x[:, :, i].mean(0).cpu().detach().numpy(), x[:, :, i].std(0).cpu().detach().numpy()
        #print(q_mean)
        hours = torch.arange(0, t + dt, dt).cpu().detach().numpy()
        axs[i].plot(hours, q_mean, label = 'Posterior mean')
        axs[i].fill_between(hours, q_mean - 2 * q_std, q_mean + 2 * q_std, alpha = 0.4, label = 'Posterior $\\mu \pm 2\sigma_x$')
        axs[i].plot(obs_model.times, obs_model.mu[i, :], linestyle = 'None', marker = 'o', label = 'Observed')
        axs[i].fill_between(obs_model.times, obs_model.mu[i, :] - 2 * obs_model.scale[:, i], obs_model.mu[i, :] + 2 * obs_model.scale[:, i], alpha = 0.4, label = 'Observation $\\mu \pm 2\sigma_y$')
        state = state_list[i]
        #axs[i].legend()
        plt.setp(axs[i], ylabel = state)
        ymin = ymin_list[i]
        ymax = ymax_list[i]
        axs[i].set_ylim([ymin, ymax])
        #plt.title(f'Approximate posterior $q(x|\\theta, y)$\nNumber of samples = {eval_batch_size}\nTimestep = {dt}\nIterations = {niter}')
    plt.xlabel('Hour')
    fig.savefig(f'net_iter_{niter}_t_{t}_dt_{dt}_batch_{batch_size}_samples_{eval_batch_size}_layers_{num_layers}_{now_string}.png', dpi = 300)

#Generate observation model.
obs_times, obs_means_noCO2, obs_error = csv_to_obs_df('y_from_x_t_5000_dt_0-01.csv', state_dim_SCON, t, obs_error_scale)
obs_model_noCO2 = ObsModel(active_device, TIMES = obs_times, DT = dt_flow, MU = obs_means_noCO2, SCALE = obs_error).to(active_device)

#Save net and ELBO files.
now = datetime.now()
now_string = now.strftime("%Y_%m_%d_%H_%M_%S")

net = torch.load('net_iter_17000_t_1400_dt_0.1_batch_1_samples_1_layers_5_2021_06_07_17_53_19.pt')
net.eval()
x, _ = net(eval_batch_size)

plot_states_post(x, obs_model_noCO2, niter, t, dt_flow, batch_size, eval_batch_size, num_layers, now_string, ymin_list = [0, 0, 0], ymax_list = [80, 2.8, 5.0])
