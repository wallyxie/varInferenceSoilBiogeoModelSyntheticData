import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
from datetime import datetime

#Torch-related imports
import torch
import torch.distributions as D
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function

#Model-specific imports
from SBM_SDE import *
from obs_and_flow import *
from training import *

def plot_elbo(elbo_hist, niter, t, dt, batch_size, eval_batch_size, num_layers, xmin = 0, ymax = None, yscale = 'linear'):
    iters = torch.arange(xmin + 1, len(elbo_hist) + 1).cpu().detach().numpy()
    plt.plot(iters, elbo_hist[xmin:])
    plt.ylim((None, ymax))
    plt.yscale(yscale)
    plt.ylabel('ELBO')
    plt.xlabel('Iteration')
    plt.title(f'ELBO history after {xmin} iterations')
    plt.savefig(f'ELBO_iter_{niter}_t_{t}_dt_{dt}_batch_{batch_size}_samples_{eval_batch_size}_layers_{num_layers}.png', dpi = 300)
    
def plot_states_post(x, obs_model, niter, t, dt, batch_size, eval_batch_size, num_layers, ymin = None, ymax = None, state_dim = 3):
    state_list = ['SOC', 'DOC', 'MBC', 'EEC']   
    fig, axs = plt.subplots(state_dim)

    for i in range(state_dim):
        q_mean, q_std = x[:, :, i].mean(0).cpu().detach().numpy(), x[:, :, i].std(0).cpu().detach().numpy()
        hours = torch.arange(0, t + dt, dt).cpu().detach().numpy()
        axs[i].plot(hours, q_mean, label = 'Posterior mean')
        axs[i].fill_between(hours, q_mean - 2 * q_std, q_mean + 2 * q_std, alpha = 0.5, label = 'Posterior $\\mu \pm 2\sigma_x$')
        axs[i].plot(obs_model.times, obs_model.mu[i, :], linestyle = 'None', marker = 'o', label = 'Observed')
        axs[i].fill_between(obs_model.times, obs_model.mu[i, :] - 2 * obs_model.scale[:, i], obs_model.mu[i, :] + 2 * obs_model.scale[:, i], alpha = 0.5, label = 'Observation $\\mu \pm 2\sigma_y$')
        state = state_list[i]
        #axs[i].legend()
        plt.setp(axs[i], ylabel = state)
        axs[i].set_ylim([ymin, ymax])
        #plt.title(f'Approximate posterior $q(x|\\theta, y)$\nNumber of samples = {eval_batch_size}\nTimestep = {dt}\nIterations = {niter}')
    plt.xlabel('Hour')
    fig.savefig(f'net_iter_{niter}_t_{t}_dt_{dt}_batch_{batch_size}_samples_{eval_batch_size}_layers_{num_layers}.png', dpi = 300)
