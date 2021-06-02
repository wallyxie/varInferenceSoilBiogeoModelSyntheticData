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

def plot_elbo(elbo_hist, num_layers, xmin = 0, ymax = None, yscale = 'linear'):
    iters = torch.arange(xmin + 1, len(elbo_hist) + 1)
    plt.plot(iters, elbo_hist[xmin:])
    plt.ylim((None, ymax))
    plt.yscale(yscale)
    plt.ylabel('ELBO')
    plt.xlabel('Iteration')
    plt.title(f'ELBO history after {xmin} iterations')
    plt.savefig(f'ELBO_t_{t}_dt_{dt_flow}_batch_{batch_size}_layers_{num_layers}_iter_{niter}.png', dpi = 300)
    
def plot_states_post(x, obs_model, num_layers, niter, dt, num_samples = 1, ymin = None, ymax = None, state_dim = 3):
    state_list = ['SOC', 'DOC', 'MBC', 'EEC']   
    fig, axs = plt.subplots(state_dim)

    for i in range(state_dim):
        q_mean, q_std = x[:, :, i].mean(0).detach(), x[:, :, i].std(0).detach()
        hours = torch.arange(0, t + dt, dt)
        axs[i].plot(hours, q_mean, label='Posterior mean')
        axs[i].fill_between(hours, q_mean - 2 * q_std, q_mean + 2 * q_std, alpha = 0.5, label = 'Posterior $\\mu \pm 2\sigma_x$')
        axs[i].plot(obs_model.times, obs_model.mu[i, :], linestyle = 'None', marker = 'o', label = 'Observed')
        axs[i].fill_between(obs_model.times, obs_model.mu[i, :] - 2 * obs_model.scale[:, i], obs_model.mu[i, :] + 2 * obs_model.scale[:, state_idx], alpha=0.5, label = 'Observation $\\mu \pm 2\sigma_y$')
        state = state_list[i]
        #axs[i].legend()
        axs[i].xlabel('Hour')
        axs[i].ylabel(state)
        axs[i].ylim((ymin, ymax))
        #plt.title(f'Approximate posterior $q(x|\\theta, y)$\nNumber of samples = {num_samples}\nTimestep = {dt}\nIterations = {niter}')
    
    fig.savefig(f'net_t_{t}_dt_{dt_flow}_batch_{batch_size}_layers_{num_layers}_iter_{niter}_{state}.png', dpi = 300)
