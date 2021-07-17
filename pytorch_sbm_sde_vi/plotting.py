import matplotlib
import matplotlib.pyplot as plt

#Torch-related imports
import torch
import numpy as np

def plot_elbo(elbo_hist, niter, piter, t, dt, batch_size, eval_batch_size, num_layers, train_lr, now_string, xmin = 0, ymax = None, yscale = 'linear'):
    iters = torch.arange(xmin + 1, len(elbo_hist) + 1).cpu().detach().numpy()
    plt.plot(iters, elbo_hist[xmin:])
    plt.ylim((None, ymax))
    plt.yscale(yscale)
    plt.ylabel('ELBO')
    plt.xlabel('Iteration')
    plt.title(f'ELBO history after {xmin} iterations')
    plt.savefig(f'ELBO_iter_{niter}_piter_{piter}_t_{t}_dt_{dt}_batch_{batch_size}_samples_{eval_batch_size}_layers_{num_layers}_lr_{train_lr}_{now_string}.png', dpi = 300)
    
def plot_states_post(x, obs_model, niter, piter, t, dt, batch_size, eval_batch_size, num_layers, train_lr, now_string, ymin_list = None, ymax_list = None, state_dim = 3):
    state_list = ['SOC', 'DOC', 'MBC', 'EEC']   
    fig, axs = plt.subplots(state_dim)

    obs_model.mu = obs_model.mu.cpu().detach().numpy()
    obs_model.scale = obs_model.scale.cpu().detach().numpy()

    for i in range(state_dim):
        q_mean, q_std = x[:, :, i].mean(0).cpu().detach().numpy(), x[:, :, i].std(0).cpu().detach().numpy()
        hours = torch.arange(0, t + dt, dt).cpu().detach().numpy()
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
    fig.savefig(f'net_iter_{niter}_piter_{piter}_t_{t}_dt_{dt}_batch_{batch_size}_samples_{eval_batch_size}_layers_{num_layers}_lr_{train_lr}_{now_string}.png', dpi = 300)

def plot_theta(p_theta, q_theta, niter, piter, t, dt, batch_size, eval_batch_size, num_layers, train_lr, now_string,
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
    num_pts = 1000
    x = torch.from_numpy(np.linspace(lower, upper, num_pts))
    pdf_prior = torch.exp(q_dist.log_prob(x)).detach()
    pdf_post = torch.exp(p_dist.log_prob(x)).detach()
    
    # Plot
    fig, axes = plt.subplots(nrows, ncols, figsize=(15,15))
    k = 0
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            if k < 14:
                ax.plot(x[:, k], pdf_prior[:, k], label='Prior $p(\\theta)$')
                ax.plot(x[:, k], pdf_post[:, k], label='Approximate posterior $q(\\theta)$')
                ax.set_xlabel(q_theta.keys[k])
                ax.set_ylabel('density')
            elif k == 14:
                handles, labels = axes[0, 0].get_legend_handles_labels()
                ax.legend(handles, labels, loc='center')
                ax.axis('off')
            else:
                fig.delaxes(axes[i, j])
            k += 1
            
    plt.tight_layout()
    fig.savefig(f'theta_iter_{niter}_piter_{piter}_t_{t}_dt_{dt}_batch_{batch_size}_samples_{eval_batch_size}_layers_{num_layers}_lr_{train_lr}_{now_string}.png', dpi = 300)
