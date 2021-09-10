import os
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import torch.distributions as D
from LogitNormal import *

plt.rcParams.update({'font.size': 16, 'lines.linewidth': 2, 'lines.markersize': 10})

def plot_theta(p_theta_file, q_theta_file, true_theta_file, fig_file,
               fig_dir='figs', ncols=4, device=torch.device('cpu')):
    # Load prior distribution
    p_dist = torch.load(p_theta_file, map_location=device)
    
    # Load posterior distribution
    q_theta = torch.load(q_theta_file, map_location=device)
    if q_theta.learn_cov:
        loc = q_theta.means
        scale_tril = D.transform_to(q_theta.dist.arg_constraints['scale_tril'])(q_theta.sds)
        lower = q_theta.lowers
        upper = q_theta.uppers
        q_joint = q_theta.dist(loc, scale_tril=scale_tril, a=lower, b=upper)
        scale = torch.diag(q_joint.covariance_matrix).sqrt()
        q_dist = RescaledLogitNormal(loc, scale, a = lower, b = upper) # marginal
    else:
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
    pdf_prior = torch.exp(p_dist.log_prob(x)).detach()
    pdf_post = torch.exp(q_dist.log_prob(x)).detach()
    
    # Plot
    num_params = len(loc)
    nrows = int(num_params / ncols) + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4))
    axes = np.atleast_2d(axes)
    k = 0
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            if k < num_params:
                key = q_theta.keys[k]
                ax.plot(x[:, k], pdf_prior[:, k], label='Prior $p(\\theta)$')
                ax.plot(x[:, k], pdf_post[:, k], label='Approximate posterior $q(\\theta)$')
                ax.axvline(true_theta[key], color='gray', label='True $\\theta$')
                ax.set_xlabel(key)
                ax.set_ylabel('density')
            elif k == num_params:
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

    # If covariance is learned, also save the correlation matrix plot
    if q_theta.learn_cov:
        # Sample from multivariate distribution
        num_samples = 100000
        theta_samples = q_joint.sample((num_samples, )) # (num_samples, num_params)

        # Calculate (empirical) correlation b/w parameters
        #corr = (q_joint.covariance_matrix / torch.outer(scale, scale)).detach()
        corr = np.corrcoef(theta_samples.T)

        # Plot correlation matrix
        plt.figure(figsize = (8, 8))
        plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(shrink=0.8)
        plt.xticks(range(num_params), labels=q_theta.keys, rotation='vertical')
        plt.yticks(range(num_params), labels=q_theta.keys)
        plt.title('Correlation between parameters')

        # Save to file
        plt.savefig('/'.join([fig_dir, 'corr_{}'.format(fig_file)]), dpi=300)

def plot_states(net_file, kf_file, fig_file, fig_dir='figs', num_samples=10,
                summarize_net=True, ymin_list=None, ymax_list=None, device=torch.device('cpu')):
    # Load net object
    net = torch.load(net_file, map_location=device)
    obs_model, state_dim, t, dt = net.obs_model, net.state_dim, net.t, net.dt
    
    # Load KalmanFilter object
    kf = torch.load(kf_file, map_location=device)
    
    # Draw samples of x
    net.device = device
    net.eval()
    x = net(num_samples)[0].detach()
    
    # Define figure and axes objects
    state_list = ['SOC', 'DOC', 'MBC', 'EEC']
    fig, axs = plt.subplots(state_dim, figsize=(15, 15))
    if ymin_list is None:
        ymin_list = [None] * state_dim
    if ymax_list is None:
        ymax_list = [None] * state_dim
    
    for i in range(state_dim):
        hours = torch.arange(0, t + dt, dt)

        # Plot observations
        color = cm.get_cmap('tab10')(2)
        axs[i].plot(obs_model.times, obs_model.mu[i, :], linestyle = 'None',
                    marker = '.', label = 'Observed', color=color)
        #axs[i].fill_between(obs_model.times, obs_model.mu[i, :] - 2 * obs_model.scale[:, i], obs_model.mu[i, :] + 2 * obs_model.scale[:, i], alpha = 0.4, label = 'Observation $\\mu \pm 2\sigma_y$')
        
        # Plot net posterior
        color = cm.get_cmap('tab10')(0)
        if summarize_net:
            #net_mean, net_sd = x[:, :, i].mean(0), x[:, :, i].std(0)
            net_left, net_median, net_right = torch.quantile(x[:, :, i], torch.tensor([0.025, 0.5, 0.975]), dim=0)
            axs[i].plot(hours, net_median, label = 'Flow mean', color=color)
            axs[i].fill_between(hours, net_left, net_right,
                            alpha = 0.4, label = 'Flow 2.5-97.5%', color=color)
        else:
            for j in range(num_samples):
                label = 'Flow sample' if j == 0 else None
                axs[i].plot(hours, x[j, :, i], color=color, alpha=0.9, label = label)

        # Plot kf posterior
        color = cm.get_cmap('tab10')(1)
        kf_mean, kf_sd = kf.mu_smooth[:, i], kf.sigma_smooth[:, i, i].sqrt()
        axs[i].plot(hours, kf_mean, label = 'Kalman mean', color=color)
        axs[i].fill_between(hours, kf_mean - 2 * kf_sd, kf_mean + 2 * kf_sd, color=color,
                            alpha = 0.4, label = 'Kalman 2.5-97.5%')
        
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
