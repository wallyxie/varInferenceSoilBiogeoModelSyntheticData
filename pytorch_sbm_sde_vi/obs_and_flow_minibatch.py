#Python-related imports
import os

#Torch imports
import torch
from torch.autograd import Function
from torch import nn
import torch.distributions as D
import torch.nn.functional as F
import torch.optim as optim

#PyData imports
import numpy as np
import pandas as pd

'''
This module contains the constituent classes for the generative flow used to represent the neural differential equations corresponding to the soil biogeochemical model SDE systems, along with the observation model class and miscellanious data processing functions.
'''

###########################################
##SYNTHETIC OBSERVATION READ-IN FUNCTIONS##
###########################################

def csv_to_obs_df(df_csv_string, dim, T, obs_error_scale):
    '''
    Takes CSV of labeled biogeochemical data observations and returns three items: 
    1) Numpy array of observation measurement times.
    2) Observations tensor including observations up to desired experiment hour threshold. 
    3) Observation error standard deviation at desired proportion of mean observation values. 
    '''
    obs_df_full = pd.read_csv(df_csv_string)
    obs_df = obs_df_full[obs_df_full['hour'] <= T]    
    obs_times = np.array(obs_df['hour'])    
    obs_means = torch.Tensor(np.array(obs_df.drop(columns = 'hour')))    
    obs_means_T = obs_means.T
    obs_error_sd =  obs_error_scale * torch.mean(obs_means_T, 1)
    obs_error_sd_re = obs_error_sd.reshape([1, dim]) #Need to reshape observation error tensor for input into ObsModel class.
    return obs_times, obs_means_T, obs_error_sd_re

##################################################
##NORMALIZING FLOW RELATED CLASSES AND FUNCTIONS##
##################################################

class LowerBound(Function):
    
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones(inputs.size()) * bound
        b = b.to(inputs.device)
        b = b.type(inputs.dtype)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None

class MaskedConv1d(nn.Conv1d):
    
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv1d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        with torch.no_grad():
            self.register_buffer('mask', self.weight.clone())
        _, _, kW = self.weight.size() # (out_cha, in_cha, kernel_size)
        self.mask.fill_(1)
        self.mask[:, :, kW // 2 + 1 * (mask_type == 'B'):] = 0 # A:[1, 0, 0] or B:[1, 1, 0]

    def forward(self, x):
        with torch.no_grad():
            self.weight *= self.mask
        return super(MaskedConv1d, self).forward(x)

class ResNetBlock(nn.Module):
    
    def __init__(self, inp_cha, out_cha, kernel = 3, stride = 1, first = False, batch_norm = False):
        super().__init__()
        self.conv1 = MaskedConv1d('A' if first else 'B', inp_cha, out_cha, kernel, stride, kernel//2, bias = False)
        self.conv2 = MaskedConv1d('B', out_cha,  out_cha, kernel, 1, kernel//2, bias = False)

        self.act1 = nn.PReLU(out_cha, init = 0.2)
        self.act2 = nn.PReLU(out_cha, init = 0.2)

        if batch_norm:
            self.bn1 = nn.BatchNorm1d(out_cha)
            self.bn2 = nn.BatchNorm1d(out_cha)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()

        if inp_cha != out_cha or stride > 1:
            self.conv_skip = MaskedConv1d('A' if first else 'B', inp_cha,  out_cha, kernel, stride, kernel//2, bias = False)
        else:
            self.conv_skip = nn.Identity()
            
        self.win = (kernel - 1) + 1*first

    def forward(self, x):
        residual = x
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x) + self.conv_skip(residual)))
        return x
    
    @property
    def window(self):
        return self.win
    
class PermutationLayer(nn.Module):
    
    def __init__(self, STATE_DIM):
        super().__init__()
        self.state_dim = STATE_DIM
        self.perm_idx = torch.randperm(STATE_DIM)

    def forward(self, x, *args, **kwargs):
        bsz, ch, w = x.shape # (batch_size, 1, state_dim * n)
        x = x.reshape(bsz, -1, self.state_dim)
        x = x[:, :, self.perm_idx]
        x = x.reshape(bsz, ch, w)
        return x, 0

class AffineLayer(nn.Module):
    
    def __init__(self, cond_inputs, kernel, num_resblocks, theta_dim, theta_cond = 'convolution', h_cha = 96):
        super().__init__()
        self.theta_cond = theta_cond
        
        network = []
        
        if theta_dim is not None and theta_cond:
            if theta_cond == 'linear':
                self.nin = nn.Sequential(nn.Linear(theta_dim+h_cha, h_cha),
                                         nn.PReLU(),
                                         nn.Linear(h_cha, h_cha),
                                         nn.PReLU(),
                                         nn.Linear(h_cha, h_cha),
                                         nn.PReLU(),
                                         nn.Linear(h_cha, 2))
            elif theta_cond == 'convolution':
                cond_inputs += theta_dim
        else:
            if theta_dim is None and theta_cond != False:
                raise Error('theta_dim is None, but theta_cond is not False. Either theta_dim needs int input, or theta_cond needs to be False.')
        
        for i in range(num_resblocks):
            if i == 0:
                network += [ResNetBlock(1+cond_inputs, h_cha, kernel=kernel, first=True)]
            else:
                network += [ResNetBlock(h_cha, h_cha, kernel=kernel)]
                
        if theta_cond == 'linear':
            network += [MaskedConv1d('B', h_cha, 2 if theta_dim is None else h_cha, kernel, 1, kernel//2)]
        else:
            network += [MaskedConv1d('B', h_cha, 2, kernel, 1, kernel//2)]
        
        self.network = nn.Sequential(*network)
        self.kernel_size = kernel
        self.alpha = nn.Parameter(torch.Tensor([0.1])) 
        self.gamma = nn.Parameter(torch.Tensor([0.]))
        
    def forward(self, x, cond_inputs, *args, **kwargs): # x.shape == (batch_size, 1, n * state_dim)
        theta = kwargs.get("theta", None)
        if theta is not None and self.theta_cond:
            if self.theta_cond == 'linear':
                output_pre = self.network(torch.cat([x, cond_inputs], 1))
                theta = theta[:, :, None].repeat(1, 1, output_pre.shape[-1])
                output_pre = torch.cat([theta, output_pre], 1).permute(0, 2, 1)
                output = self.nin(output_pre).permute(0, 2, 1)  
            elif self.theta_cond == 'convolution':    
                theta = theta[:, :, None].repeat(1, 1, x.shape[-1])
                output = self.network(torch.cat([x, cond_inputs, theta], 1))
        else:
            output = self.network(torch.cat([x, cond_inputs], 1))
        mu, sigma = torch.chunk(self.alpha*output, 2, 1) # (batch_size, 1, n * state_dim)
        sigma = (self.gamma*sigma).exp()
        x = self.alpha*mu + sigma * x # (batch_size, 1, n * state_dim)
        return x, -torch.log(sigma) # each of shape (batch_size, 1, n * state_dim)
    
    @property
    def window(self):
        win = self.kernel_size//2
        for l in self.network[:-1]:
            win += l.window
        return win
        
class SoftplusLayer(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()
    
    def forward(self, x, *args, **kwargs):
        y = self.softplus(x)
        return y, -torch.log(-torch.expm1(-y))

class BatchNormLayer(nn.Module):
    
    def __init__(self, num_inputs, momentum = 1e-2, eps = 1e-5, affine = True):
        super(BatchNormLayer, self).__init__()

        self.log_gamma = nn.Parameter(torch.rand(num_inputs)) if affine else torch.zeros(num_inputs)
        self.beta = nn.Parameter(torch.rand(num_inputs)) if affine else torch.zeros(num_inputs)
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, inputs, *args, **kwargs):
        inputs = inputs.squeeze(1) # (batch_size, n * state_dim)
        if self.training:
            # Compute mean and var across batch
            self.batch_mean = inputs.mean()
            self.batch_var = (inputs - self.batch_mean).pow(2).mean() + self.eps

            self.running_mean.mul_(self.momentum)
            self.running_var.mul_(self.momentum)

            self.running_mean.add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.add_(self.batch_var.data * (1 - self.momentum))

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        
        x_hat = (inputs - mean) / var.sqrt() # (batch_size, n * state_dim)
        y = torch.exp(self.log_gamma) * x_hat + self.beta # (batch_size, n * state_dim)
        ildj = -self.log_gamma + 0.5 * torch.log(var) # (n * state_dim, )

        return y[:, None, :], ildj[None, None, :]
    
class SDEFlowMinibatch(nn.Module):

    def __init__(self, DEVICE, OBS_MODEL_MINIBATCH, STATE_DIM, T, N, THETA_DIM,
                 OTHER_INPUTS = None, FIX_THETA_DICT = None,
                 NUM_LAYERS = 5, KERNEL_SIZE = 3, NUM_RESBLOCKS = 2, 
                 POSITIVE = True, THETA_COND = 'convolution'):
        super().__init__()
        
        self.device = DEVICE
        self.state_dim = STATE_DIM
        self.t = T
        self.dt = OBS_MODEL_MINIBATCH.dt
        self.n = N
        self.theta_dim = THETA_DIM
        
        #Transform time-related tensors into shape for conditional inputs and rescale timestamp smaller per Tom's suggestion.
        timestamp = torch.linspace(-1, 1, self.n)[None].repeat(self.state_dim, 1).transpose(1, 0).reshape(1, -1) # (1, n * state_dim)
        
        #future observation count
        # NOTE: This currently assumes a regular time gap between observations!
        steps_bw_obs = OBS_MODEL_MINIBATCH.idx[1] - OBS_MODEL_MINIBATCH.idx[0]
        reps = [self.state_dim * steps_bw_obs] * (len(OBS_MODEL_MINIBATCH.idx) - 1)
        future_reps = torch.tensor([self.state_dim] + reps).to(self.device)
        future_obs = OBS_MODEL_MINIBATCH.mu.repeat_interleave(future_reps, -1) # (obs_dim, n * state_dim)

        #past observation count
        past_reps = torch.tensor(reps + [self.state_dim]).to(self.device)
        past_obs = OBS_MODEL_MINIBATCH.mu.repeat_interleave(past_reps, -1) # (obs_dim, n * state_dim)

        #Combine time cond_inputs.
        cond_inputs_list = [timestamp, future_obs, past_obs]

        if OTHER_INPUTS is not None:
            cond_inputs_list.append(OTHER_INPUTS)

        if FIX_THETA_DICT is not None:
            fix_theta_tensor = torch.tensor(list(FIX_THETA_DICT.values())) # (num_fixed_params, )
            fix_theta_tensor = fix_theta_tensor[:, None].repeat(1, N * self.state_dim) # (num_fixed_params, n * state_dim)
            cond_inputs_list.append(fix_theta_tensor)

        self.cond_inputs = torch.cat(cond_inputs_list, 0) # (num_features, n * state_dim)
        self.n_cond_inputs = self.cond_inputs.shape[0] # number of static features (don't change every iter)
        self.num_layers = NUM_LAYERS
        self.kernel_size = KERNEL_SIZE
        self.num_resblocks = NUM_RESBLOCKS
        
        self.positive = POSITIVE
        self.theta_cond = THETA_COND
        
        self.scale = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)

        layers = []
        for i in range(self.num_layers):
            layers += [AffineLayer(self.n_cond_inputs, self.kernel_size, self.num_resblocks, self.theta_dim, self.theta_cond)]
            layers += [PermutationLayer(self.state_dim)]
            layers += [BatchNormLayer(1)] # WARNING: This might be less effective (seems to currently work)
            
        layers.pop(-1)
        layers.pop(-1)
        if self.positive:
            layers += [SoftplusLayer()]
        self.layers = nn.ModuleList(layers)
        
    def forward(self, bsz, lidx, ridx, *args, **kwargs):
        # bsz - batch size
        # lidx - left index
        # ridx - left index
        buffer = self.state_dim*(ridx - lidx)
        lidx = max((lidx*self.state_dim) - self.window, 0)
        ridx = ridx * self.state_dim
        
        theta = kwargs.get("theta", None)

        base_dist = D.normal.Normal(loc = 0., scale = LowerBound.apply(self.scale, 1e-6))
        eps = base_dist.sample([bsz, self.state_dim * self.n]).to(self.device)
        log_prob = base_dist.log_prob(eps).permute(0, 2, 1)[:, :, lidx:ridx]
        
        cond_inputs = self.cond_inputs[None, :, lidx:ridx].repeat(bsz, 1, 1)
        eps = eps.permute(0, 2, 1)[:, :, lidx:ridx]

        ildjs = []
        for layer in self.layers:
            eps, ildj = layer(eps, cond_inputs, theta=theta)
            ildjs.append(ildj)
            
        for ildj in ildjs:
            log_prob += ildj
        
        eps = eps[:, :, -buffer:]
        log_prob = log_prob[:, :, -buffer:]
            
        #return eps.reshape(bsz, -1, self.state_dim).transpose(2, 1), log_prob
        return eps.reshape(bsz, -1, self.state_dim), log_prob
    
    @property
    def window(self):
        win = 0
        for l in self.layers:
            if isinstance(l, AffineLayer):
                win += l.window 
        win = int(np.ceil(win/self.state_dim) * self.state_dim)
        return win

###################################################
##OBSERVATION MODEL RELATED CLASSES AND FUNCTIONS##
###################################################

class ObsModel(nn.Module):
    def __init__(self, TIMES, DT, MU, SCALE):
        super().__init__()
        self.times = TIMES # (n_obs, )
        self.dt = DT
        self.idx = self.get_idx(TIMES, DT)        
        self.mu = torch.Tensor(MU) # (obs_dim, n_obs)
        self.scale = SCALE # (1, obs_dim)
        self.obs_dim = self.mu.shape[0]
        
    def forward(self, x, theta):
        obs_ll = D.normal.Normal(self.mu.permute(1, 0), self.scale).log_prob(x[:, self.idx, :])
        return torch.sum(obs_ll, [-1, -2]).mean()

    def get_idx(self, TIMES, DT):
        return torch.as_tensor((TIMES / DT)).long() #list((TIMES / DT).astype(int))
    
    def plt_dat(self):
        return self.mu, self.times

class ObsModelMinibatch(ObsModel):
    def __init__(self, TIMES, DT, MU, SCALE):
        super().__init__(TIMES, DT, MU, SCALE)

        # NOTE: Assumes regular obs_every interval starting from the 0th index,
        # otherwise not sure how to convert from lidx/ridx to obs_lidx/obs_lidx
        self.obs_every = self.idx[1] - self.idx[0]
        
    def forward(self, x, theta, lidx=0, ridx=None):
        obs_lidx = int(torch.ceil(lidx / self.obs_every))
        if ridx is None:
            obs_ridx = self.mu.shape[1]
        else:
            obs_ridx = int(torch.ceil(ridx / self.obs_every))
        
        loc = self.mu.permute(1, 0)[obs_lidx:obs_ridx, :] # (n_obs_minibatch, obs_dim)
        idx = self.idx[obs_lidx:obs_ridx] - lidx
        #print(torch.tensor(idx) - lidx)
        obs_ll = D.normal.Normal(loc, self.scale).log_prob(x[:, idx, :]) # (batch_size, n_obs_minibatch, obs_dim)
        return torch.sum(obs_ll, [-1, -2]).mean() # scalar

########################
##MISC MODEL DEBUGGING##
########################

class ModelSaver():
    "rolling window model saver"
    
    def __init__(self, win=5, save_dir="./model_saves/", cleanup=True):
        #save last win models
        self.win = win
        self.save_dir = save_dir
        
        #WARNING: will wipe models in save_dir - use a diff save_dir for every experiment
        if cleanup:
            [os.remove(f"{self.save_dir}{n}") for n in os.listdir(self.save_dir) if n.split(".")[-1] == "pt"]
    
    def save(self, models, train_iter):
        saved_models = [n for n in os.listdir(self.save_dir) if n.split(".")[-1] == "pt"]
        if len(saved_models) >= self.win:
            self._delete(saved_models)
        torch.save(models, f"{self.save_dir}model_{train_iter}.pt")
            
    def _delete(self, saved_models):
        mod_idx = np.array([int(f.split(".")[0]) for f in [f.split("_")[1] for f in saved_models]]).min()
        del_fname = f"{self.save_dir}model_{mod_idx}.pt"
        os.remove(del_fname)
