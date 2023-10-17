import argparse
import time
import math
import numpy as np
import pandas as pd
import torch

import pyro
from pyro.distributions import Normal, TorchDistribution
from pyro.infer import MCMC, NUTS, HMC
from pyro.infer.autoguide.initialization import init_to_sample, init_to_uniform

from SBM_SDE_classes_minibatch import *
import LogitNormal
import TruncatedNormal as TN

class RescaledLogitNormal(LogitNormal.RescaledLogitNormal, TorchDistribution):
    pass

class TruncatedNormal(TN.TruncatedNormal, TorchDistribution):
    pass

def csv_to_obs_df(df_csv_string, dim, T, obs_error_scale):
    '''
    Takes CSV of labeled biogeochemical data observations and returns three items: 
    1) Numpy array of observation measurement times.
    2) Observations tensor including observations up to desired experiment hour threshold. 
    3) Observation error standard deviation at desired proportion of mean observation values. 
    '''
    obs_df_full = pd.read_csv(df_csv_string)
    obs_df = obs_df_full[obs_df_full['hour'] <= T]    
    obs_times = torch.Tensor(obs_df['hour'])    
    obs_means = torch.Tensor(np.array(obs_df.drop(columns = 'hour')))   
    obs_means_T = obs_means.T
    obs_error_sd =  obs_error_scale * torch.mean(obs_means_T, 1)
    obs_error_sd_re = obs_error_sd.reshape([1, dim]) #Need to reshape observation error tensor for input into ObsModel class.
    return obs_times, obs_means_T, obs_error_sd_re

# (loc_theta, scale_theta, a_theta, b_theta) should only include ones that aren't fixed
def model(SBM_SDE, T, dt, state_dim, param_names, I_S, I_D, temp, obs_every,
          loc_theta, scale_theta, a_theta, b_theta, loc_x0, scale_x0, scale_y,
          y=None, fix_theta_dict=None):
    # Draw theta
    with pyro.plate('theta_plate', len(param_names)):
        theta = pyro.sample('theta', RescaledLogitNormal(loc_theta, scale_theta, a_theta, b_theta))
        theta_dict = {param_names[i]: theta[i] for i in range(len(param_names))}
        if fix_theta_dict:
            theta_dict = {**theta_dict, **fix_theta_dict}
    
    # Draw x_0
    with pyro.plate('x_0_plate', state_dim):
        x = pyro.sample('x_0', Normal(loc_x0, scale_x0))
    
    # Draw x_t
    x_obs = []
    #obs_idx_set = set([int(i) for i in obs_idx])
    #print(obs_idx_set)
    for t in pyro.markov(range(T + 1)):
        if t > 0:
            alpha = SBM_SDE.calc_drift(x, theta_dict, I_S[t], I_D[t], temp[t])
            beta = SBM_SDE.calc_diffusion_sqrt(x, theta_dict, diffusion_matrix=False)
            with pyro.plate('x_{}_plate'.format(t), state_dim):
                #x = pyro.sample('x_{}'.format(t), Normal(loc=x + alpha*dt, scale=beta*math.sqrt(dt)))
                x = pyro.sample('x_{}'.format(t), Normal(loc=x + alpha*dt, scale=beta*math.sqrt(dt)))
        #if t in obs_idx_set:
        if t % obs_every == 0:
            x_obs.append(x)
    
    # Draw y
    x_obs = torch.stack(x_obs, dim=0)
    CO2 = SBM_SDE.calc_CO2(x_obs, theta_dict, temp[::obs_every].reshape(-1, 1))
    x_with_CO2 = torch.cat((x_obs, CO2), dim=-1) # (T_obs, obs_dim)
    y0_plate = pyro.plate('y_t_plate', T//obs_every + 1, dim=-2)
    y1_plate = pyro.plate('y_d_plate', state_dim + 1, dim=-1)
    with y0_plate, y1_plate:
        pyro.sample('y', Normal(x_with_CO2, scale_y), obs=y) # (T_obs, obs_dim)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-samples", nargs="?", default=200, type=int)
    parser.add_argument("--num-chains", nargs="?", default=5, type=int)
    parser.add_argument("--warmup-steps", nargs="?", default=100, type=int)
    parser.add_argument("--step-size", nargs="?", default=1, type=float)
    parser.add_argument("--adapt-step-size", nargs="?", default=1, type=int)
    parser.add_argument("--jit-compile", nargs="?", default=1, type=int)
    parser.add_argument("--seed", nargs="?", default=1, type=int)
    parser.add_argument("--save-every", nargs="?", default=10, type=int)
    args = parser.parse_args()
    return args

class Logger:
    def __init__(self, log_dir, num_samples, save_every=10):
        self.t0 = time.process_time()
        self.total_time = 0
        self.times = []
        self.save_every = save_every
        self.num_samples = num_samples
        self.log_dir = log_dir
        
    def log(self, kernel, samples, stage, i):
        if stage == 'sample':
            self.times.append(time.process_time() - self.t0)
            self.total_time = self.times[-1]
            self.samples.append(samples)

            if i % self.save_every == 0 or i == self.num_samples:
                # Save log
                progress_dict = {'kernel': kernel,
                                 'samples': torch.stack(self.samples),
                                 'times': self.times}
                torch.save(progress_dict, '{}/iter{}.pt'.format(self.log_dir, i))
    
                # Clear log
                self.times = []
                self.samples = []

def run(args, model_params, in_filenames, out_filenames):
    T, dt, obs_CO2, state_dim, obs_error_scale, \
        temp_ref, temp_rise, model_type, diffusion_type, device = model_params
    obs_file, p_theta_file, x0_file = in_filenames
    out_dir, samples_file, diagnostics_file, args_file = out_filenames

    # Load data
    N = int(T / dt)
    T_span = torch.linspace(0, T, N + 1).to(device)
    obs_ndim = state_dim + 1

    print('Loading data from', obs_file)
    obs_times, obs_vals, obs_errors = csv_to_obs_df(obs_file, obs_ndim, T, obs_error_scale)
    obs_vals = obs_vals.T
    y_dict = {int(t): v for t, v in zip(obs_times.to(device), obs_vals.to(device))}
    assert y_dict[0].shape == (state_dim + 1, )
    
    # Load hyperparameters of theta
    theta_hyperparams = torch.load(p_theta_file)
    param_names = list(theta_hyperparams.keys())
    theta_hyperparams_list = list(zip(*(theta_hyperparams[k] for k in param_names))) # unzip theta hyperparams from dictionary values into individual lists
    loc_theta, scale_theta, a_theta, b_theta = torch.tensor(theta_hyperparams_list).to(device)
    assert loc_theta.shape == scale_theta.shape == a_theta.shape == b_theta.shape == (len(param_names), )

    # Load parameters of x_0
    loc_x0 = torch.load(x0_file).to(device)
    scale_x0 = obs_error_scale.to(device) * loc_x0
    assert loc_x0.shape == scale_x0.shape == (state_dim, )

    # Load parameters of y
    scale_y = obs_errors.to(device)
    assert obs_errors.shape == (1, state_dim + 1)

    hyperparams = {
        'loc_theta': loc_theta, 
        'scale_theta': scale_theta, 
        'a_theta': a_theta, 
        'b_theta': b_theta, 
        'loc_x0': loc_x0, 
        'scale_x0': scale_x0, 
        'scale_y': scale_y 
    }

    # Obtain temperature forcing function.
    temp = temp_gen(T_span, temp_ref, temp_rise).to(device)
    
    # Obtain SOC and DOC pool litter input vectors for use in flow SDE functions.
    I_S = i_s(T_span).to(device) #Exogenous SOC input function
    I_D = i_d(T_span).to(device) #Exogenous DOC input function
    
    # Instantiate SBM_SDE object
    SBM_SDE = model_type(T_span, I_S, I_D, temp, temp_ref, diffusion_type)
    print('Using model', SBM_SDE.__class__.__name__, SBM_SDE.DIFFUSION_TYPE)
    
    # Instantiate MCMC object
    kernel = NUTS(model,
                  step_size=args.step_size,
                  adapt_step_size=bool(args.adapt_step_size),
                  init_strategy=init_to_sample,
                  jit_compile=bool(args.jit_compile), ignore_jit_warnings=True)
    logger = Logger(out_dir, args.num_samples, args.save_every)
    mcmc = MCMC(kernel,
                num_samples=args.num_samples,
                warmup_steps=args.warmup_steps,
                num_chains=args.num_chains,
                hook_fn=logger.log())
    
    # Run MCMC and record runtime per iteration
    torch.manual_seed(args.seed)
    model_args = (SBM_SDE, T, dt, state_dim, param_names, I_S, I_D, temp,
                  int(obs_times[1] - obs_times[0]))
    model_kwargs = {'y': obs_vals.to(device)}
    model_kwargs.update(hyperparams)
    mcmc.run(*model_args, **model_kwargs)
    print('Total time:', logger.total_time, 'seconds')
    
    # Save results
    print('Saving MCMC samples and diagnostics to', out_dir)
    samples = mcmc.get_samples(group_by_chain=True)
    diagnostics = mcmc.diagnostics()
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    torch.save(samples, samples_file)
    torch.save(diagnostics, diagnostics_file)
    torch.save((model_args, model_kwargs), args_file)
    #torch.save((model_args, model_kwargs, times, samples, diagnostics), out_filename)

    # Print MCMC diagnostics summary
    mcmc.summary()
