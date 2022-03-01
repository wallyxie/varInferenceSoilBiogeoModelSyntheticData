#Python-related imports
import os.path

#PyData imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#Torch-related imports
import torch
import torch.distributions as D

#Module imports
from SBM_SDE_classes import *
from TruncatedNormal import *
from LogitNormal import *

def plot_elbo(elbo_hist, niter, warmup_iter, t, dt, batch_size, eval_batch_size, num_layers, train_lr, decay_step, warmup_lr, sd_scale, plots_folder, now_string, xmin = 0, ymax = None, yscale = 'linear'):
    iters = torch.arange(xmin + 1, len(elbo_hist) + 1).detach().cpu().numpy()
    plt.plot(iters, elbo_hist[xmin:])
    plt.ylim((None, ymax))
    plt.yscale(yscale)
    plt.ylabel('ELBO')
    plt.xlabel('Iteration')
    plt.title(f'ELBO history after {xmin} iterations')
    plt.savefig(os.path.join(plots_folder, f'ELBO_iter_{niter}_warmup_{warmup_iter}_t_{t}_dt_{dt}_batch_{batch_size}_samples_{eval_batch_size}_layers_{num_layers}_lr_{train_lr}_decay_step_{decay_step}_warmup_lr_{warmup_lr}_sd_scale_{sd_scale}_{now_string}.png'), dpi = 300)
    
def plot_states_post(x, q_theta, obs_model, SBM_SDE_CLASS, niter, warmup_iter, t, dt, batch_size, eval_batch_size, num_layers, train_lr, decay_step, warmup_lr, sd_scale, plots_folder, now_string, FIX_THETA_DICT = None, LEARN_CO2 = False, ymin_list = None, ymax_list = None):

    state_list = []

    if x.size(-1) == 3 and not LEARN_CO2:
        state_list = ['SOC', 'DOC', 'MBC']
    elif x.size(-1) == 3 and LEARN_CO2:
        state_list = ['SOC', 'DOC', 'MBC', 'CO2']
    elif x.size(-1) == 4 and not LEARN_CO2:
        state_list = ['SOC', 'DOC', 'MBC', 'EEC']
    elif x.size(-1) == 4 and LEARN_CO2:
        state_list = ['SOC', 'DOC', 'MBC', 'EEC', 'CO2']
    else:
        raise Exception('Matching condition does not exist with x.size() and LEARN_CO2 status.')

    if LEARN_CO2:
        q_theta_sample_dict, _, _, _ = q_theta(x.size(0))
        if FIX_THETA_DICT:
            q_theta_sample_dict = {**q_theta_sample_dict, **FIX_THETA_DICT}
        x = SBM_SDE_CLASS.add_CO2(x, q_theta_sample_dict) #Add CO2 to x tensor if CO2 is being fit.

    fig, axs = plt.subplots(x.size(-1))

    obs_model.mu = obs_model.mu.detach().cpu().numpy()
    obs_model.scale = obs_model.scale.detach().cpu().numpy()

    for i in range(x.size(-1)):
        q_mean, q_std = x[:, :, i].mean(0).detach().cpu().numpy(), x[:, :, i].std(0).detach().cpu().numpy()
        hours = torch.arange(0, t + dt, dt).detach().cpu().numpy()
        axs[i].plot(obs_model.times, obs_model.mu[i, :], linestyle = 'None', marker = '.', label = 'Observed')
        axs[i].fill_between(obs_model.times, obs_model.mu[i, :] - 2 * obs_model.scale[:, i], obs_model.mu[i, :] + 2 * obs_model.scale[:, i], alpha = 0.4, label = 'Observation $\\mu \pm 2\sigma_y$')
        axs[i].plot(hours, q_mean, label = 'Posterior mean')
        axs[i].fill_between(hours, q_mean - 2 * q_std, q_mean + 2 * q_std, alpha = 0.4, label = 'Posterior $\\mu \pm 2\sigma_x$')
        state = state_list[i]
        #axs[i].legend()
        plt.setp(axs[i], ylabel = state)
        ymin = ymin_list[i] if ymin_list else None
        ymax = ymax_list[i] if ymax_list else None
        axs[i].set_ylim([ymin, ymax])
        #plt.title(f'Approximate posterior $q(x|\\theta, y)$\nNumber of samples = {eval_batch_size}\nTimestep = {dt}\nIterations = {niter}')
    plt.xlabel('Hour')
    plt.tight_layout()
    fig.set_size_inches(20, 15)
    fig.savefig(os.path.join(plots_folder, f'net_iter_{niter}_warmup_{warmup_iter}_t_{t}_dt_{dt}_batch_{batch_size}_samples_{eval_batch_size}_layers_{num_layers}_lr_{train_lr}_decay_step_{decay_step}_warmup_lr_{warmup_lr}_sd_scale_{sd_scale}_{now_string}.png'), dpi = 300)

def plot_states_NN(x, params_dict, obs_model, SBM_SDE_CLASS, niter, warmup_iter, t, dt, batch_size, eval_batch_size, num_layers, train_lr, decay_step, warmup_lr, sd_scale, plots_folder, now_string, LEARN_CO2 = False, ymin_list = None, ymax_list = None):

    state_list = []

    if x.size(-1) == 3 and not LEARN_CO2:
        state_list = ['SOC', 'DOC', 'MBC']
    elif x.size(-1) == 3 and LEARN_CO2:
        state_list = ['SOC', 'DOC', 'MBC', 'CO2']
    elif x.size(-1) == 4 and not LEARN_CO2:
        state_list = ['SOC', 'DOC', 'MBC', 'EEC']
    elif x.size(-1) == 4 and LEARN_CO2:
        state_list = ['SOC', 'DOC', 'MBC', 'EEC', 'CO2']
    else:
        raise Exception('Matching condition does not exist with x.size() and LEARN_CO2 status.')

    if LEARN_CO2:
        x = SBM_SDE_CLASS.add_CO2(x, params_dict) #Add CO2 to x tensor if CO2 is being fit.

    fig, axs = plt.subplots(x.size(-1))

    obs_model.mu = obs_model.mu.detach().cpu().numpy()
    obs_model.scale = obs_model.scale.detach().cpu().numpy()

    for i in range(x.size(-1)):
        q_mean, q_std = x[:, :, i].mean(0).detach().cpu().numpy(), x[:, :, i].std(0).detach().cpu().numpy()
        hours = torch.arange(0, t + dt, dt).detach().cpu().numpy()
        axs[i].plot(obs_model.times, obs_model.mu[i, :], linestyle = 'None', marker = '.', label = 'Observed')
        axs[i].fill_between(obs_model.times, obs_model.mu[i, :] - 2 * obs_model.scale[:, i], obs_model.mu[i, :] + 2 * obs_model.scale[:, i], alpha = 0.4, label = 'Observation $\\mu \pm 2\sigma_y$')
        axs[i].plot(hours, q_mean, label = 'Posterior mean')
        axs[i].fill_between(hours, q_mean - 2 * q_std, q_mean + 2 * q_std, alpha = 0.4, label = 'Posterior $\\mu \pm 2\sigma_x$')
        state = state_list[i]
        #axs[i].legend()
        plt.setp(axs[i], ylabel = state)
        ymin = ymin_list[i] if ymin_list else None
        ymax = ymax_list[i] if ymax_list else None
        axs[i].set_ylim([ymin, ymax])
        #plt.title(f'Approximate posterior $q(x|\\theta, y)$\nNumber of samples = {eval_batch_size}\nTimestep = {dt}\nIterations = {niter}')
    plt.xlabel('Hour')
    #plt.tight_layout()
    fig.set_size_inches(20, 15)
    fig.savefig(os.path.join(plots_folder, f'net_iter_{niter}_warmup_{warmup_iter}_t_{t}_dt_{dt}_batch_{batch_size}_samples_{eval_batch_size}_layers_{num_layers}_lr_{train_lr}_decay_step_{decay_step}_warmup_lr_{warmup_lr}_sd_scale_{sd_scale}_{now_string}.png'), dpi = 300)

def plot_theta(p_theta, q_theta, true_theta, niter, warmup_iter, t, dt, batch_size, eval_batch_size, num_layers, train_lr, decay_step, warmup_lr, sd_scale, plots_folder, now_string, ncols=4):
    # Prior distribution object
    p_dist = q_theta.dist(p_theta.loc.detach().cpu(), torch.max(p_theta.scale, torch.tensor(1e-8)).detach().cpu(), p_theta.a.detach().cpu(), p_theta.b.detach().cpu()) 

    # Posterior distribution object
    if q_theta.learn_cov:
        loc = q_theta.means.detach().cpu()
        scale_tril = D.transform_to(q_theta.dist.arg_constraints['scale_tril'])(q_theta.sds.detach().cpu())
        lower = q_theta.lowers.detach().cpu()
        upper = q_theta.uppers.detach().cpu()
        q_joint = q_theta.dist(loc, scale_tril=scale_tril, a=lower, b=upper)
        scale = torch.diag(q_joint.covariance_matrix).sqrt()
        q_dist = RescaledLogitNormal(loc, scale, a = lower, b = upper) # marginal
    else:
        loc = q_theta.means.detach().cpu()
        scale = torch.max(q_theta.sds, torch.tensor(1e-8)).detach().cpu()
        lower = q_theta.lowers.detach().cpu()
        upper = q_theta.uppers.detach().cpu()
        q_dist = q_theta.dist(loc, scale, a = lower, b = upper)
    
    # Compute prior and posterior densities at points x
    num_pts = 10000
    x = torch.zeros([num_pts, loc.size(0)]).detach().cpu() #Examining densities as we move through distribution supports. So torch.Size([bins, parameters]) is desired size of x.
    x0 = torch.min(q_dist.mean - 4 * q_dist.stddev, p_dist.mean.detach().cpu() - 4 * p_dist.stddev.detach().cpu())
    x0 = torch.max(x0, lower).detach().cpu()

    x1 = torch.max(q_dist.mean + 4 * q_dist.stddev, p_dist.mean.detach().cpu() + 4 * p_dist.stddev.detach().cpu())
    x1 = torch.min(x1, upper).detach().cpu()

    for param_index in range(0, loc.size(0)):
        x[:, param_index] = torch.linspace(x0[param_index], x1[param_index], num_pts).detach().cpu()

    pdf_prior = torch.exp(p_dist.log_prob(x)).detach().cpu()
    pdf_post = torch.exp(q_dist.log_prob(x)).detach().cpu()

    # #Find appropriate plotting range of x based on pdf density concentration (where pdf > 1 for prior and post).    
    # prior_first_one_indices = torch.zeros(loc.size(0))
    # prior_last_one_indices = torch.zeros(loc.size(0))
    # post_first_one_indices = torch.zeros(loc.size(0))
    # post_last_one_indices = torch.zeros(loc.size(0))
    # for param_index in range(0, loc.size(0)):
    #     prior_geq_ones = pdf_prior[:, param_index] >= 1e-6 #Find pdf values >= 1 in prior.
    #     prior_cumsum = prior_geq_ones.cumsum(axis = -1) #Cumsum over all true values.
    #     prior_min_index = prior_cumsum.min(0).indices
    #     prior_max_index = prior_cumsum.max(0).indices
    #     prior_first_one_indices[param_index] = prior_min_index
    #     prior_last_one_indices[param_index] = prior_max_index
    #     post_geq_ones = pdf_post[:, param_index] >= 1e-6 #Find pdf values >= 1 in posterior.
    #     post_cumsum = post_geq_ones.cumsum(axis = -1) #Cumsum over all true values.
    #     post_min_index = post_cumsum.min(0).indices
    #     post_max_index = post_cumsum.max(0).indices 
    #     post_first_one_indices[param_index] = post_min_index
    #     post_last_one_indices[param_index] = post_max_index

    # Plot
    num_params = len(loc)
    nrows = int(num_params / ncols) + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = np.atleast_2d(axes)
    param_index = 0
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            if param_index < num_params:
                key = q_theta.keys[param_index]
                ax.plot(x[:, param_index], pdf_prior[:, param_index], label='Prior $p(\\theta)$')
                ax.plot(x[:, param_index], pdf_post[:, param_index], label='Approximate posterior $q(\\theta)$')
                ax.axvline(true_theta[key], color='gray', label='True $\\theta$')
                ax.set_xlabel(key)
                ax.set_ylabel('Density')
                ax.ticklabel_format(style='sci', scilimits=(-2,4), axis='both', useMathText='True')
            elif param_index == num_params:
                handles, labels = axes[0, 0].get_legend_handles_labels()
                ax.legend(handles, labels, loc='center')
                ax.axis('off')
            else:
                fig.delaxes(axes[i, j])
            param_index += 1
            
    plt.tight_layout()
    fig.savefig(os.path.join(plots_folder, f'theta_iter_{niter}_warmup_{warmup_iter}_t_{t}_dt_{dt}_batch_{batch_size}_samples_{eval_batch_size}_layers_{num_layers}_lr_{train_lr}_decay_step_{decay_step}_warmup_lr_{warmup_lr}_sd_scale_{sd_scale}_{now_string}.png'), dpi = 300)
