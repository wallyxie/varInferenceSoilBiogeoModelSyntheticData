import matplotlib
import matplotlib.pyplot as plt

#Torch-related imports
import torch
import numpy as np

def plot_elbo(elbo_hist, niter, piter, t, dt, batch_size, eval_batch_size, num_layers, train_lr, sd_scale, now_string, xmin = 0, ymax = None, yscale = 'linear'):
    iters = torch.arange(xmin + 1, len(elbo_hist) + 1).detach().cpu().numpy()
    plt.plot(iters, elbo_hist[xmin:])
    plt.ylim((None, ymax))
    plt.yscale(yscale)
    plt.ylabel('ELBO')
    plt.xlabel('Iteration')
    plt.title(f'ELBO history after {xmin} iterations')
    plt.savefig(f'ELBO_iter_{niter}_piter_{piter}_t_{t}_dt_{dt}_batch_{batch_size}_samples_{eval_batch_size}_layers_{num_layers}_lr_{train_lr}_sd_scale_{sd_scale}_{now_string}.png', dpi = 300)
    
def plot_states_post(x, obs_model, niter, piter, t, dt, batch_size, eval_batch_size, num_layers, train_lr, sd_scale, now_string, ymin_list = None, ymax_list = None, state_dim = 3):
    state_list = ['SOC', 'DOC', 'MBC', 'EEC']   
    fig, axs = plt.subplots(state_dim)

    obs_model.mu = obs_model.mu.detach().cpu().numpy()
    obs_model.scale = obs_model.scale.detach().cpu().numpy()

    for i in range(state_dim):
        q_mean, q_std = x[:, :, i].mean(0).detach().cpu().numpy(), x[:, :, i].std(0).detach().cpu().numpy()
        hours = torch.arange(0, t + dt, dt).detach().cpu().numpy()
        axs[i].plot(obs_model.times, obs_model.mu[i, :], linestyle = 'None', marker = '.', label = 'Observed')
        axs[i].fill_between(obs_model.times, obs_model.mu[i, :] - 2 * obs_model.scale[:, i], obs_model.mu[i, :] + 2 * obs_model.scale[:, i], alpha = 0.4, label = 'Observation $\\mu \pm 2\sigma_y$')
        axs[i].plot(hours, q_mean, label = 'Posterior mean')
        axs[i].fill_between(hours, q_mean - 2 * q_std, q_mean + 2 * q_std, alpha = 0.4, label = 'Posterior $\\mu \pm 2\sigma_x$')
        state = state_list[i]
        #axs[i].legend()
        plt.setp(axs[i], ylabel = state)
        ymin = ymin_list[i]
        ymax = ymax_list[i]
        axs[i].set_ylim([ymin, ymax])
        #plt.title(f'Approximate posterior $q(x|\\theta, y)$\nNumber of samples = {eval_batch_size}\nTimestep = {dt}\nIterations = {niter}')
    plt.xlabel('Hour')
    fig.savefig(f'net_iter_{niter}_piter_{piter}_t_{t}_dt_{dt}_batch_{batch_size}_samples_{eval_batch_size}_layers_{num_layers}_lr_{train_lr}_sd_scale_{sd_scale}_{now_string}.png', dpi = 300)

def plot_theta(p_theta, q_theta, niter, piter, t, dt, batch_size, eval_batch_size, num_layers, train_lr, sd_scale, now_string,
               nrows=4, ncols=4):
    # Prior distribution object
    p_dist = p_theta

    # Posterior distribution object
    loc = q_theta.means
    scale = torch.max(q_theta.sds, torch.tensor(1e-6))
    lower = q_theta.lowers
    upper = q_theta.uppers
    q_dist = q_theta.dist(loc, scale, a = lower, b = upper)
    
    # Compute prior and posterior densities at points x
    num_pts = 10000000
    x = torch.zeros([num_pts, loc.size(0)]) #Examining densities as we move through distribution supports. So torch.Size([bins, parameters]) is desired size of x.
    for param_index in range(0, loc.size(0)):
        x[:, param_index] = torch.linspace(lower[param_index], upper[param_index], num_pts)
    pdf_prior = torch.exp(q_dist.log_prob(x)).detach()
    pdf_post = torch.exp(p_dist.log_prob(x)).detach()

    #Find appropriate plotting range of x based on pdf density concentration (where pdf > 1 for prior and post).    
    prior_first_one_indices = torch.zeros(loc.size(0))
    prior_last_one_indices = torch.zeros(loc.size(0))
    post_first_one_indices = torch.zeros(loc.size(0))
    post_last_one_indices = torch.zeros(loc.size(0))
    for param_index in range(0, loc.size(0)):
        prior_geq_ones = pdf_prior[:, param_index] >= 2e-5 #Find pdf values >= 1 in prior.
        prior_cumsum = prior_geq_ones.cumsum(axis = -1) #Cumsum over all true values.
        prior_min_index = prior_cumsum.min(0).indices
        prior_max_index = prior_cumsum.max(0).indices
        prior_first_one_indices[param_index] = prior_min_index
        prior_last_one_indices[param_index] = prior_max_index
        post_geq_ones = pdf_post[:, param_index] >= 2e-5 #Find pdf values >= 1 in posterior.
        post_cumsum = post_geq_ones.cumsum(axis = -1) #Cumsum over all true values.
        post_min_index = post_cumsum.min(0).indices
        post_max_index = post_cumsum.max(0).indices 
        post_first_one_indices[param_index] = post_min_index
        post_last_one_indices[param_index] = post_max_index

    # Plot
    fig, axes = plt.subplots(nrows, ncols, figsize=(15,15))
    param_index = 0
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            if param_index < 14:
                ax.plot(x[int(prior_first_one_indices[param_index]): int(prior_last_one_indices[param_index]), param_index].detach().cpu().numpy(), pdf_prior[int(prior_first_one_indices[param_index]): int(prior_last_one_indices[param_index]), param_index].detach().cpu().numpy(), label='Prior $p(\\theta)$')
                ax.plot(x[int(post_first_one_indices[param_index]): int(post_last_one_indices[param_index]), param_index].detach().cpu().numpy(), pdf_post[int(post_first_one_indices[param_index]): int(post_last_one_indices[param_index]), param_index].detach().cpu().numpy(), label='Approximate posterior $q(\\theta)$')
                ax.set_xlabel(q_theta.keys[param_index])
                ax.set_ylabel('Density')
            elif param_index == 14:
                handles, labels = axes[0, 0].get_legend_handles_labels()
                ax.legend(handles, labels, loc='center')
                ax.axis('off')
            else:
                fig.delaxes(axes[i, j])
            param_index += 1
            
    plt.tight_layout()
    fig.savefig(f'theta_iter_{niter}_piter_{piter}_t_{t}_dt_{dt}_batch_{batch_size}_samples_{eval_batch_size}_layers_{num_layers}_lr_{train_lr}_sd_scale_{sd_scale}_{now_string}.png', dpi = 300)
