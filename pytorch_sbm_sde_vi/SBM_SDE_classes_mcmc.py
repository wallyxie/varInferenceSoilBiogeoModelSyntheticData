import math
import numpy as np
import pandas as pd
import torch

import pyro
from pyro.distributions import Normal, TorchDistribution
import LogitNormal
import TruncatedNormal as TN

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

def arrhenius_temp_dep(parameter, temp, Ea, TEMP_REF) -> torch.Tensor:
    '''
    Arrhenius temperature dependence function.
    Accepts input parameter as torch.Tensor or Python Number type.
    Accepts Ea as torch.Tensor type only.
    0.008314 is the gas constant. Temperatures are in K.
    Returns a tensor of transformed parameter value(s).    
    '''
    decayed_parameter = parameter * torch.exp(-Ea / 0.008314 * (1 / temp - 1 / TEMP_REF))
    return decayed_parameter

class RescaledLogitNormal(LogitNormal.RescaledLogitNormal, TorchDistribution):
    pass

class TruncatedNormal(TN.TruncatedNormal, TorchDistribution):
    pass

class SCON(nn.Module):
    def __init__(self, T, dt, state_dim, temp_ref, diffusion_type):
        super().__init__()
        self.T = T
        self.dt = dt
        self.state_dim = state_dim
        self.obs_dim = state_dim + 1
        self.temp_ref temp_ref
        self.diffusion_type = diffusion_type
        
        self.N = int(T / dt) + 1
        self.times = torch.linspace(0, T, self.N)
        self.I_S = self.calc_i_S()
        self.I_D = self.calc_i_D()
        self.temp = self.calc_temps()

    def calc_temps(self):
        '''
        Temperature function to force soil biogeochemical models.
        Accepts input time(s) t in torch.Tensor type.
        This particular temperature function assumes soil temperatures will increase by TEMP_REF over the next 80 years.    
        Returns a tensor of one or more temperatures in K given t.
        '''
        t = self.times
        temps = self.temp_ref + (self.temp_rise * t) / (80 * 24 * 365) + 10 * torch.sin((2 * np.pi / 24) * t) + 10 * torch.sin((2 * np.pi / (24 * 365)) * t)
        return temps

    def calc_i_S(self) -> torch.Tensor:
        '''
        This is the endogenous SOC litter input function.
        '''
        return 0.001 + 0.0005 * torch.sin((2 * np.pi / (24 * 365)) * self.times)
    
    def calc_i_D(self) -> torch.Tensor:
        '''
        This is the endogenous DOC litter input function.
        '''
        return 0.0001 + 0.00005 * torch.sin((2 * np.pi / (24 * 365)) * self.times)

    def load_data(self, obs_error_scale, obs_file, p_theta_file, x0_file):
        print('Loading data from', obs_file)
        obs_times, obs_vals, obs_errors = csv_to_obs_df(obs_file, self.obs_dim, self.T, obs_error_scale)
        obs_vals = obs_vals.T
        #y_dict = {int(t): v for t, v in zip(obs_times.to(device), obs_vals.to(device))}
        #assert y_dict[0].shape == (state_dim + 1, )

        # Load parameters of y
        self.scale_y = obs_errors #.to(device)
        assert obs_errors.shape == (1, state_dim + 1)
        self.obs_every = int(obs_times[1] - obs_times[0])
        
        # Load hyperparameters of theta
        # (loc_theta, scale_theta, a_theta, b_theta) should only include ones that aren't fixed
        theta_hyperparams = torch.load(p_theta_file)
        self.param_names = list(theta_hyperparams.keys())
        theta_hyperparams_list = list(zip(*(theta_hyperparams[k] for k in param_names))) # unzip theta hyperparams from dictionary values into individual lists
        loc_theta, scale_theta, a_theta, b_theta = torch.tensor(theta_hyperparams_list) #.to(device)
        assert loc_theta.shape == scale_theta.shape == a_theta.shape == b_theta.shape == (len(param_names), )
        self.p_theta = Normal(loc_theta, scale_theta, a_theta, b_theta)
    
        # Load parameters of x_0
        loc_x0 = torch.load(x0_file) #.to(device)
        scale_x0 = obs_error_scale * loc_x0
        assert loc_x0.shape == scale_x0.shape == (state_dim, )
        self.p_x0 = RescaledLogitNormal(loc_x0, scale_x0)
    
        #hyperparams = {
        #    'loc_theta': loc_theta, 
        #    'scale_theta': scale_theta, 
        #    'a_theta': a_theta, 
        #    'b_theta': b_theta, 
        #    'loc_x0': loc_x0, 
        #    'scale_x0': scale_x0, 
        #    'scale_y': scale_y 
        #}

        return obs_vals

    def model(self, y=None, fix_theta_dict=None):
        # Draw theta
        with pyro.plate('theta_plate', len(self.param_names)):
            theta = pyro.sample('theta', self.p_theta)
            theta_dict = {param_names[i]: theta[i] for i in range(len(self.param_names))}
            if fix_theta_dict:
                theta_dict = {**theta_dict, **fix_theta_dict}
            weight_alpha, bias_alpha, weight_beta, bias_beta, weight_obs = self.calc_params(theta_dict)

        # Draw x_0
        with pyro.plate('x_0_plate', self.state_dim):
            x = pyro.sample('x_0', self.p_x0)
        
        # Draw x_t
        for t in pyro.markov(range(1, self.N)):
            alpha = self.calc_drift(x, weight_alpha[t], bias_alpha[t])
            beta = self.calc_diffusion(x, weight_beta)
            with pyro.plate('x_{}_plate'.format(t), self.state_dim):
                x = pyro.sample('x_{}'.format(t), Normal(loc=x + alpha*dt, scale=math.sqrt(beta*dt)))
        
        # Draw y
        num_obs = len(self.times[::self.obs_every])
        loc_y = torch.matmul(weight_obs, x[::self.obs_every].unsqueeze(-1)).squeeze() # (num_obs, obs_dim)
        y0_plate = pyro.plate('y_t_plate', num_obs, dim=-2)
        y1_plate = pyro.plate('y_d_plate', self.state_dim + 1, dim=-1)
        
        with y0_plate, y1_plate:
            pyro.sample('y', Normal(loc_y, self.scale_y), obs=y) # (T_obs, obs_dim)
    
    def calc_params(self, theta):
        # Apply temperature-dependent transformation to k_*_ref
        k_S = arrhenius_temp_dep(theta['k_S_ref'], self.temp, theta['Ea_S'], self.temp_ref) # (N, )
        k_D = arrhenius_temp_dep(theta['k_D_ref'], self.temp, theta['Ea_D'], self.temp_ref) # (N, )
        k_M = arrhenius_temp_dep(theta['k_M_ref'], self.temp, theta['Ea_M'], self.temp_ref) # (N, )
        
        # Drift params
        A0 = torch.stack([-k_S, params['a_DS'] * k_D, params['a_M'] * params['a_MSC'] * k_M])
        A1 = torch.stack([params['a_SD'] * k_S, -(params['u_M'] + k_D), params['a_M'] * (1 - params['a_MSC']) * k_M])
        A2 = torch.stack([torch.zeros(N + 1), torch.ones(N + 1) * params['u_M'], -k_M])
        weight_alpha = torch.stack([A0, A1, A2]).permute((2, 0, 1)) # (N, 3, 3)
        bias_alpha = torch.stack([self.I_S, self.I_D, torch.zeros(N + 1)]).T # (N, 3)
        
        # Diffusion params
        if self.diffusion_type == 'C':
            weight_beta = 0
            bias_beta = torch.diag(torch.tensor([theta['c_SOC'],
                                                 theta['c_DOC'],
                                                 theta['c_MBC']])) # (3, 3)
        elif self.diffusion_type == 'SS':
            weight_beta = torch.diag(torch.tensor([theta['s_SOC'],
                                                   theta['s_DOC'],
                                                   theta['s_MBC']])) # (3, 3)
            bias_beta = 0
        else:
            raise ValueError, "Unknown diffusion type"
    
        # Observation params
        num_obs = len(self.times[::self.obs_every]) #T//obs_every + 1
        C0 = torch.eye(self.state_ndim).unsqueeze(0) * torch.ones((num_obs, 1, 1)) # (N, 3, 3)
        C1 = torch.stack([(1 - params['a_SD']) * k_S[::self.obs_every],
                          (1 - params['a_DS']) * k_D[::self.obs_every],
                          (1 - params['a_M']) * k_M][::self.obs_every]).unsqueeze(0).permute((2, 0, 1)) # (N, 1, 3) 
        weight_obs = torch.cat((C0, C1), dim=1) # (num_obs, 4, 3)
    
        return weight_alpha, bias_alpha, weight_beta, bias_beta, weight_obs
    
    def calc_drift(self, x, weight_alpha, bias_alpha):
        return weight_alpha @ x + bias_alpha
    
    def calc_diffusion(self, x, weight_beta, bias_beta):
        return torch.clamp(weight_beta @ x + bias_beta, min=1e-6, max=None)
