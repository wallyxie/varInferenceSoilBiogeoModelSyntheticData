import os
import torch
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 16, 'lines.linewidth': 3, 'lines.markersize': 10})

def plot_theta(p_theta_file, q_theta_file, true_theta_file, fig_file,
               fig_dir='figs', nrows=4, ncols=4, device=torch.device('cpu')):
    # Load prior distribution
    p_dist = torch.load(p_theta_file, map_location=device)
    
    # Load posterior distribution
    q_theta = torch.load(q_theta_file, map_location=device)
    loc = q_theta.means
    scale = torch.max(q_theta.sds, torch.tensor(1e-6))
    lower = q_theta.lowers
    upper = q_theta.uppers
    q_dist = q_theta.dist(loc, scale, a = lower, b = upper)
    
    # Load true theta
    true_theta = torch.load(true_theta_file, map_location=device)

    # Define plot boundaries
    #print(q_dist, q_dist.loc, q_dist.scale, q_dist.mean, q_dist.stddev)
    #print(p_dist, p_dist.loc, p_dist.scale, p_dist.mean, p_dist.stddev)

    x0 = torch.min(q_dist.mean - 4*q_dist.stddev, p_dist.mean - 4*p_dist.stddev)
    x0 = torch.max(x0, lower).detach()
    #print(x0)
    
    x1 = torch.max(q_dist.mean + 4*q_dist.stddev, p_dist.mean + 4*p_dist.stddev)
    x1 = torch.min(x1, upper).detach()
    #print(x1)
    
    # Compute prior and posterior densities at points x
    num_pts = 1000
    x = torch.from_numpy(np.linspace(x0, x1, num_pts))
    pdf_prior = torch.exp(q_dist.log_prob(x)).detach()
    pdf_post = torch.exp(p_dist.log_prob(x)).detach()
    
    # Plot
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 15))
    k = 0
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            if k < 14:
                key = q_theta.keys[k]
                ax.plot(x[:, k], pdf_prior[:, k], label='Prior $p(\\theta)$')
                ax.plot(x[:, k], pdf_post[:, k], label='Approximate posterior $q(\\theta)$')
                ax.axvline(true_theta[key], color='gray', label='True $\\theta$')
                ax.set_xlabel(key)
                ax.set_ylabel('density')
            elif k == 14:
                handles, labels = axes[0, 0].get_legend_handles_labels()
                ax.legend(handles, labels, loc='center')
                ax.axis('off')
            else:
                fig.delaxes(axes[i, j])
            k += 1  
    plt.tight_layout()

    # Save to file
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig('/'.join([fig_dir, fig_file]), dpi=300)

def plot_states(net_file, kf_file, fig_file, fig_dir='figs', num_samples=10,
                ymin_list=None, ymax_list=None, device=torch.device('cpu')):
    # Load net object
    net = torch.load(net_file, map_location=device)
    obs_model, state_dim, t, dt = net.obs_model, net.state_dim, net.t, net.dt
    
    # Load KalmanFilter object
    kf = torch.load(kf_file, map_location=device)
    
    # Draw samples of x
    net.device = device
    net.eval()
    x, _ = net(num_samples)
    
    # Define figure and axes objects
    state_list = ['SOC', 'DOC', 'MBC', 'EEC']
    fig, axs = plt.subplots(state_dim, figsize=(15, 15))
    if ymin_list is None:
        ymin_list = [None] * state_dim
    if ymax_list is None:
        ymax_list = [None] * state_dim
    
    for i in range(state_dim):
        net_mean, net_sd = x[:, :, i].mean(0).detach(), x[:, :, i].std(0).detach()
        kf_mean, kf_sd = kf.mu_smooth[:, i], kf.sigma_smooth[:, i, i].sqrt()
        hours = torch.arange(0, t + dt, dt)
        
        # Plot net posterior
        axs[i].plot(hours, net_mean, label = 'Flow mean')
        axs[i].fill_between(hours, net_mean - 2 * net_sd, net_mean + 2 * net_sd,
                            alpha = 0.4, label = 'Flow $\\mu \pm 2\sigma$')
        
        # Plot kf posterior
        axs[i].plot(hours, kf_mean, label = 'Kalman mean')
        axs[i].fill_between(hours, kf_mean - 2 * kf_sd, kf_mean + 2 * kf_sd,
                            alpha = 0.4, label = 'Kalman $\\mu \pm 2\sigma$')
        
        # Plot observations
        axs[i].plot(obs_model.times, obs_model.mu[i, :], linestyle = 'None', marker = '.', label = 'Observed')
        #axs[i].fill_between(obs_model.times, obs_model.mu[i, :] - 2 * obs_model.scale[:, i], obs_model.mu[i, :] + 2 * obs_model.scale[:, i], alpha = 0.4, label = 'Observation $\\mu \pm 2\sigma_y$')
        
        state = state_list[i]
        axs[i].set_ylabel(state) #plt.setp(axs[i], ylabel = state)
        ymin = ymin_list[i]
        ymax = ymax_list[i]
        axs[i].set_ylim([ymin, ymax])
        #plt.title(f'Approximate posterior $q(x|\\theta, y)$\nNumber of samples = {eval_batch_size}\nTimestep = {dt}\nIterations = {niter}')
    
    axs[0].legend()
    plt.xlabel('Hour')

    # Save to file
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig('/'.join([fig_dir, fig_file]), dpi=300)
