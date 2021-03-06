import matplotlib
import matplotlib.pyplot as plt

#Torch-related imports
import torch

def plot_elbo(elbo_hist, niter, piter, t, dt, batch_size, eval_batch_size, num_layers, now_string, xmin = 0, ymax = None, yscale = 'linear'):
    iters = torch.arange(xmin + 1, len(elbo_hist) + 1).cpu().detach().numpy()
    plt.plot(iters, elbo_hist[xmin:])
    plt.ylim((None, ymax))
    plt.yscale(yscale)
    plt.ylabel('ELBO')
    plt.xlabel('Iteration')
    plt.title(f'ELBO history after {xmin} iterations')
    plt.savefig(f'ELBO_iter_{niter}_piter_{piter}_t_{t}_dt_{dt}_batch_{batch_size}_samples_{eval_batch_size}_layers_{num_layers}_{now_string}.png', dpi = 300)
    
def plot_states_post(x, obs_model, niter, piter, t, dt, batch_size, eval_batch_size, num_layers, now_string, ymin_list = None, ymax_list = None, state_dim = 3):
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
    fig.savefig(f'net_iter_{niter}_piter_{piter}_t_{t}_dt_{dt}_batch_{batch_size}_samples_{eval_batch_size}_layers_{num_layers}_{now_string}.png', dpi = 300)
