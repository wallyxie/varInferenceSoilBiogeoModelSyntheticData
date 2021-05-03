from matplotlib import pyplot as plt

def plot_elbo(elbo_hist, xmin=None, ymax=None):
    plt.plot(elbo_hist)
    plt.xlim((xmin, None))
    plt.ylim((None, ymax))
    plt.ylabel('ELBO')
    plt.xlabel('Iteration')

def plot_post(net, x0, obs_model, state_idx=0, num_samples=20,
              ymin=None, ymax=None):
    x, _ = net(num_samples)
    x0 = x0[(None,) * 2].repeat(num_samples, 1, 1)
    x = torch.cat((x0, x), 1)
    
    q_mean, q_std = x[:, :, state_idx].mean(0).detach(), x[:, :, state_idx].std(0).detach()
    hours = torch.arange(0, t + 1, dt)
    plt.plot(hours, q_mean)
    plt.fill_between(hours, q_mean - 2*q_std, q_mean + 2*q_std, alpha=0.5)
    plt.plot(obs_model.times, obs_model.mu[state_idx, :], linestyle='None', marker='o')
    
    plt.xlabel('Hour')
    plt.ylabel(['SOC', 'DOC', 'MBC'][state_idx])
    plt.ylim((ymin, ymax))
    plt.title('Approximate posterior $q(x|\\theta, y)$')
