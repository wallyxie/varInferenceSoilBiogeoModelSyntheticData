import math
import numpy as np
import pandas as pd
import torch
import pyro
from pyro.distributions import Normal, TorchDistribution

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
def model(SBM_SDE, T, dt, state_dim, param_names, I_S, I_D, temp, obs_idx,
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
        x = pyro.sample('x_0', Normal(loc_x0, scale_x0, a=0, b=1000))
    
    # Draw x_t
    x_obs = []
    obs_idx_set = set([int(i) for i in obs_idx])
    #print(obs_idx_set)
    for t in pyro.markov(range(T + 1)):
        if t > 0:
            alpha = SBM_SDE.calc_drift(x, theta_dict, I_S[t], I_D[t], temp[t])
            beta = SBM_SDE.calc_diffusion_sqrt(x, theta_dict, diffusion_matrix=False)
            with pyro.plate('x_{}_plate'.format(t), state_dim):
                #x = pyro.sample('x_{}'.format(t), Normal(loc=x + alpha*dt, scale=beta*math.sqrt(dt)))
                x = pyro.sample('x_{}'.format(t), Normal(loc=x + alpha*dt, scale=beta*math.sqrt(dt), a=0, b=1000))
        if t in obs_idx_set:
            x_obs.append(x)
    
    # Draw y
    x_obs = torch.stack(x_obs, dim=0)
    CO2 = SBM_SDE.calc_CO2(x_obs, theta_dict, temp[obs_idx].reshape(-1, 1))
    x_with_CO2 = torch.cat((x_obs, CO2), dim=-1) # (T_obs, obs_dim)
    y0_plate = pyro.plate('y_t_plate', len(obs_idx), dim=-2)
    y1_plate = pyro.plate('y_d_plate', state_dim + 1, dim=-1)
    with y0_plate, y1_plate:
        pyro.sample('y', Normal(x_with_CO2, scale_y), obs=y) # (T_obs, obs_dim)

    #print(x_obs)
