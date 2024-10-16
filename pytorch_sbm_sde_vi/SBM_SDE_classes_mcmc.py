import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import distributions as D
from scipy.interpolate import splrep, BSpline

import pyro
from pyro.distributions import Normal, TorchDistribution
import LogitNormal
#import TruncatedNormal as TN

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

#class TruncatedNormal(TN.TruncatedNormal, TorchDistribution):
#    pass

class SCON(nn.Module):
    def __init__(self, T, dt, state_dim, temp_ref, temp_rise, diffusion_type):
        super().__init__()
        self.T = T
        self.dt = dt
        self.state_dim = state_dim
        self.obs_dim = state_dim + 1
        self.temp_ref = temp_ref
        self.temp_rise = temp_rise
        self.diffusion_type = diffusion_type
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.N = int(T / dt) + 1
        self.times = torch.linspace(0, T, self.N).to(self.device)
        self.I_S = self.calc_i_S().to(self.device)
        self.I_D = self.calc_i_D().to(self.device)
        self.temp = self.calc_temps().to(self.device)
        self.to(self.device)

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

    def load_data(self, obs_error_scale, obs_file, p_theta_file, x0_file, fix_theta_file=None):
        print('Loading data from', obs_file)
        obs_times, obs_vals, obs_errors = csv_to_obs_df(obs_file, self.obs_dim, self.T, obs_error_scale)
        obs_vals = obs_vals.T.to(self.device)
        #y_dict = {int(t): v for t, v in zip(obs_times.to(self.device), obs_vals.to(self.device))}
        #assert y_dict[0].shape == (state_dim + 1, )

        # Load parameters of y
        self.scale_y = obs_errors.squeeze().to(self.device)
        assert self.scale_y.shape == (self.obs_dim, )
        self.obs_every = int(obs_times[1] - obs_times[0])
        
        # Load p(theta)
        theta_hyperparams = torch.load(p_theta_file)
        self.param_names = list(theta_hyperparams.keys())
        theta_hyperparams_list = list(zip(*(theta_hyperparams[k] for k in self.param_names))) # unzip theta hyperparams from dictionary values into individual lists
        loc_theta, scale_theta, a_theta, b_theta = torch.tensor(theta_hyperparams_list).to(self.device)
        assert loc_theta.shape == scale_theta.shape == a_theta.shape == b_theta.shape == (len(self.param_names), )
        self.p_theta = RescaledLogitNormal(loc_theta, scale_theta, a_theta, b_theta)
        
        # Load fix_theta_dict (if provided)
        if fix_theta_file:
            fix_theta_dict = {k: torch.tensor(v) for k, v in torch.load(fix_theta_file).items()}
            self.param_names = list(fix_theta_dict.keys())
        else:
            fix_theta_dict = None

        # Load parameters of x_0
        loc_x0 = torch.load(x0_file).to(self.device)
        scale_x0 = obs_error_scale * loc_x0
        assert loc_x0.shape == scale_x0.shape == (self.state_dim, )
        self.p_x0 = Normal(loc_x0, scale_x0)
    
        return (obs_vals, fix_theta_dict)

    def model(self, y=None, fix_theta_dict=None):
        # Draw theta
        with pyro.plate('theta_plate', len(self.param_names)):
            theta = pyro.sample('theta', self.p_theta)
            theta_dict = {self.param_names[i]: theta[i] for i in range(len(self.param_names))}
            if fix_theta_dict:
                theta_dict = {**theta_dict, **fix_theta_dict}
            weight_alpha, bias_alpha, weight_beta, bias_beta, weight_obs = self.calc_params(theta_dict)

        # Draw x_0
        x_all = []
        with pyro.plate('x_0_plate', self.state_dim):
            x = pyro.sample('x_0', self.p_x0)
            x_all.append(x)

        # Draw x_t
        for t in pyro.markov(range(1, self.N)):
            #if t == 1:
            #    print(weight_alpha[t].shape, x.shape, bias_alpha[t].shape)
            #    print(weight_beta.shape, x.shape, bias_beta.shape)
            alpha = self.calc_drift(x, weight_alpha[t], bias_alpha[t])
            beta = self.calc_diffusion(x, weight_beta, bias_beta)
            with pyro.plate('x_{}_plate'.format(t), self.state_dim):
                loc = x + alpha*self.dt
                scale = torch.sqrt(beta*self.dt)
            #    if t == 1: print(loc.shape, scale.shape)
                x = pyro.sample('x_{}'.format(t), Normal(loc=loc, scale=scale))
                x_all.append(x)
        x_all = torch.stack(x_all, dim=0)
        assert x_all.shape == (self.N, self.state_dim)
        
        # Draw y
        num_obs = len(self.times[::self.obs_every])
        # weight_obs.shape == (num_obs, self.obs_dim, self.state_dim)
        # x_obs.shape == (num_obs, state_dim)
        loc_y = torch.matmul(weight_obs, x_all[::self.obs_every].unsqueeze(-1)).squeeze() # (num_obs, obs_dim)
        y0_plate = pyro.plate('y_t_plate', num_obs, dim=-2)
        y1_plate = pyro.plate('y_d_plate', self.state_dim + 1, dim=-1)
        
        assert loc_y.shape == (num_obs, self.obs_dim) and y.shape == (num_obs, self.obs_dim)
        with y0_plate, y1_plate:
            pyro.sample('y', Normal(loc_y, self.scale_y), obs=y) # (T_obs, obs_dim)
    
    def calc_params(self, theta):
        # Apply temperature-dependent transformation to k_*_ref
        k_S = arrhenius_temp_dep(theta['k_S_ref'], self.temp, theta['Ea_S'], self.temp_ref) # (N, )
        k_D = arrhenius_temp_dep(theta['k_D_ref'], self.temp, theta['Ea_D'], self.temp_ref) # (N, )
        k_M = arrhenius_temp_dep(theta['k_M_ref'], self.temp, theta['Ea_M'], self.temp_ref) # (N, )
        
        # Drift params
        A0 = torch.stack([-k_S, theta['a_DS'] * k_D, theta['a_M'] * theta['a_MSC'] * k_M])
        A1 = torch.stack([theta['a_SD'] * k_S, -(theta['u_M'] + k_D), theta['a_M'] * (1 - theta['a_MSC']) * k_M])
        A2 = torch.stack([torch.zeros(self.N, device=self.device), torch.ones(self.N, device=self.device) * theta['u_M'], -k_M])
        weight_alpha = torch.stack([A0, A1, A2]).permute((2, 0, 1)) # (N, 3, 3)
        bias_alpha = torch.stack([self.I_S, self.I_D, torch.zeros(self.N, device=self.device)]).T # (N, 3)
        assert weight_alpha.shape == (self.N, self.state_dim, self.state_dim)
        assert bias_alpha.shape == (self.N, self.state_dim)

        # Diffusion params
        if self.diffusion_type == 'C':
            weight_beta = torch.zeros((self.state_dim, self.state_dim), device=self.device)
            bias_beta = torch.tensor([theta['c_SOC'],
                                      theta['c_DOC'],
                                      theta['c_MBC']]).to(self.device) # (3, )
        elif self.diffusion_type == 'SS':
            weight_beta = torch.diag(torch.tensor([theta['s_SOC'],
                                                   theta['s_DOC'],
                                                   theta['s_MBC']], dtype=torch.float64)).to(self.device) # (3, 3)
            bias_beta = torch.zeros(self.state_dim, device=self.device)
        else:
            raise ValueError("Unknown diffusion type")
        assert weight_beta.shape == (self.state_dim, self.state_dim)
        assert bias_beta.shape == (self.state_dim, )
    
        # Observation params
        num_obs = len(self.times[::self.obs_every]) #T//obs_every + 1
        C0 = torch.eye(self.state_dim, device=self.device).unsqueeze(0) * torch.ones((num_obs, 1, 1), device=self.device) # (N, 3, 3)
        C1 = torch.stack([(1 - theta['a_SD']) * k_S[::self.obs_every],
                          (1 - theta['a_DS']) * k_D[::self.obs_every],
                          (1 - theta['a_M']) * k_M[::self.obs_every]]).unsqueeze(0).permute((2, 0, 1)) # (N, 1, 3) 
        weight_obs = torch.cat((C0, C1), dim=1) # (num_obs, 4, 3)
        assert weight_obs.shape == (num_obs, self.obs_dim, self.state_dim)

        return weight_alpha, bias_alpha, weight_beta, bias_beta, weight_obs
    
    def calc_drift(self, x, weight_alpha, bias_alpha):
        return torch.matmul(weight_alpha, x) + bias_alpha
    
    def calc_diffusion(self, x, weight_beta, bias_beta):
        beta = torch.matmul(weight_beta, x) + bias_beta
        return torch.clamp(beta, min=1e-6, max=None)

    def sde_log_prob(self, x, theta_dict): # log p(x|theta)
        # x.shape == (T, state_dim), theta.shape == (num_params, )
        weight_alpha, bias_alpha, weight_beta, bias_beta, weight_obs = self.calc_params(theta_dict)
        # weight_alpha.shape == (T, state_dim, state_dim)
        # bias_alpha.shape == (T, state_dim)
        # weight_beta.shape == (state_dim, state_dim)
        # bias_beta.shape == (state_dim, )
        # weight_obs.shape == (num_obs, obs_dim, state_dim)

        #x = x.unsqueeze(-1)                              # (T, state_dim, 1)
        bias_alpha = bias_alpha.unsqueeze(-1)            # (T, state_dim, 1)
        weight_beta = weight_beta.unsqueeze(0)           # (1, state_dim, state_dim)
        bias_beta = bias_beta.unsqueeze(0).unsqueeze(-1) # (1, state_dim, 1)

        alpha = self.calc_drift(x[:-1].unsqueeze(-1), weight_alpha[1:], bias_alpha[1:]).squeeze()  # (T-1, state_dim)
        beta = self.calc_diffusion(x[:-1].unsqueeze(-1), weight_beta, bias_beta).squeeze()         # (T-1, state_dim)
        loc = x[:-1] + alpha*self.dt
        scale = torch.sqrt(beta*self.dt)
        p_x = D.normal.Normal(loc=loc, scale=scale)

        # Compute log p(x|theta) = log p(x_1:T|x0, theta) + log p(x0|theta)
        log_prob = p_x.log_prob(x[1:, :]).sum(0)     # log p(x_1:T|x0, theta)
        log_prob += self.p_x0.log_prob(x[0])         # log p(x0|theta)
        #print('Shapes', log_prob.shape, weight_obs.shape) # (state_dim, )

        return log_prob.sum(), weight_obs # log_obs.shape == scalar

    def obs_log_prob(self, x, y, theta_dict, weight_obs): # log p(y|x, theta)
        # weight_obs.shape == (num_obs, self.obs_dim, self.state_dim)
        # x_obs.shape == (num_obs, state_dim)
        loc_y = torch.matmul(weight_obs, x[::self.obs_every].unsqueeze(-1)).squeeze() # (num_obs, obs_dim)
        num_obs = len(self.times[::self.obs_every])
        assert loc_y.shape == (num_obs, self.obs_dim) and y.shape == (num_obs, self.obs_dim)
        p_y = D.normal.Normal(loc=loc_y, scale=self.scale_y)
        log_p_y = p_y.log_prob(y) # (num_obs, obs_dim)
        #print(log_p_y.shape)

        return log_p_y.sum()

    def log_prob(self, x, y, logit_theta, fix_theta_dict=None):
        if fix_theta_dict is None:
            # log p(theta)
            theta = self.p_theta.sigmoid(logit_theta)
            assert torch.all(self.p_theta.a < theta) and torch.all(theta < self.p_theta.b)
            param_log_prob = self.p_theta.log_prob(theta).sum()
            theta_dict = {self.param_names[i]: theta[i] for i in range(len(self.param_names))}
        else:
            theta_dict = fix_theta_dict

        # log p(x|theta)
        sde_log_prob, weight_obs = self.sde_log_prob(x, theta_dict)
    
        # log p(y|x, theta)
        obs_log_prob = self.obs_log_prob(x, y, theta_dict, weight_obs)
        
        # log joint = log p(theta) + log p(x|theta) + log p(y|x, theta)
        if fix_theta_dict is None:
            return param_log_prob + sde_log_prob + obs_log_prob
        else:
            return sde_log_prob + obs_log_prob

    def add_CO2(self, x, theta):
        # Apply temperature-dependent transformation to k_*_ref
        k_S = arrhenius_temp_dep(theta['k_S_ref'], self.temp, theta['Ea_S'], self.temp_ref) # (N, )
        k_D = arrhenius_temp_dep(theta['k_D_ref'], self.temp, theta['Ea_D'], self.temp_ref) # (N, )
        k_M = arrhenius_temp_dep(theta['k_M_ref'], self.temp, theta['Ea_M'], self.temp_ref) # (N, )

        # Observation params
        C0 = torch.eye(self.state_dim, device=self.device).unsqueeze(0) * torch.ones((self.N, 1, 1), device=self.device) # (N, 3, 3)
        C1 = torch.stack([(1 - theta['a_SD']) * k_S,
                          (1 - theta['a_DS']) * k_D,
                          (1 - theta['a_M']) * k_M]).unsqueeze(0).permute((2, 0, 1)) # (N, 1, 3) 
        weight_obs = torch.cat((C0, C1), dim=1) # (N, 4, 3)
        assert weight_obs.shape == (self.N, self.obs_dim, self.state_dim)

        return torch.matmul(weight_obs, x.unsqueeze(-1)).squeeze() # (N, obs_dim)

    def sample(self, y, fix_theta_dict=None):
        # Sample theta
        if fix_theta_dict:
            theta_dict = fix_theta_dict
            theta = torch.stack([theta_dict[k] for k in self.param_names])
        else:
            theta = self.p_theta.sample()
            theta_dict = {self.param_names[i]: theta[i] for i in unfixed_param_indices}
        #if fix_theta_dict:
        #    theta_dict = {**theta_dict, **fix_theta_dict}
        weight_alpha, bias_alpha, weight_beta, bias_beta, weight_obs = self.calc_params(theta_dict)

        # Sample x_0
        x_all = []
        x = self.p_x0.sample()
        x_all.append(x)

        # Sample x_t
        for t in range(1, self.N):
            alpha = self.calc_drift(x, weight_alpha[t], bias_alpha[t])
            beta = self.calc_diffusion(x, weight_beta, bias_beta)
            loc = x + alpha*self.dt
            scale = torch.sqrt(beta*self.dt)
            p_x = D.normal.Normal(loc=loc, scale=scale)
            x = p_x.sample()
            x_all.append(x)
        x_all = torch.cat(x_all, dim=0)
        assert x_all.shape == (self.N * self.state_dim, )

        # Transform theta to real space
        logit_theta = self.p_theta.logit(theta)
            
        return torch.cat((logit_theta, x_all)).to(self.device)

    def smooth_init(self, y, fix_theta_dict=None):
        obs_times = self.times[::self.obs_every].cpu()
        obs_vals = y.cpu()
        x_all = []
        for i in range(self.state_dim):
            tck = splrep(obs_times, obs_vals[:, i], s=self.T)
            xvals = np.arange(0, self.T + self.dt, self.dt)
            x_all.append(torch.tensor(BSpline(*tck)(xvals)))

        return torch.stack(x_all, dim=-1).reshape(-1).to(self.device)
        