#Torch-related imports
import torch
from torch.autograd import Function
from torch import nn
import torch.distributions as D
import torch.nn.functional as F
import torch.optim as optim

#PyData imports
import numpy as np
import pandas as pd

#Python-related imports
import os

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
        b = torch.ones(inputs.size()).to(inputs.device) * bound
        b = b.type(inputs.dtype)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b).to(inputs.device)

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
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kW = self.weight.size() # (out_cha, in_cha, kernel_size)
        self.mask.fill_(1)
        self.mask[:, :, kW // 2 + 1 * (mask_type == 'B'):] = 0 # [1, 0, 0] or [1, 1, 0]

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv1d, self).forward(x)

class ResNetBlock(nn.Module):
    
    def __init__(self, inp_cha, out_cha, stride = 1, first = True, batch_norm = True):
        super().__init__()
        self.conv1 = MaskedConv1d('A' if first else 'B', inp_cha,  out_cha, 3, stride, 1, bias = False)
        self.conv2 = MaskedConv1d('B', out_cha,  out_cha, 3, 1, 1, bias = False)

        self.act1 = nn.PReLU(out_cha, init = 0.2)
        self.act2 = nn.PReLU(out_cha, init = 0.2)

        if batch_norm:
            self.bn1 = nn.BatchNorm1d(out_cha)
            self.bn2 = nn.BatchNorm1d(out_cha)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()

        # If dimensions change, transform shortcut with a conv layer
        if inp_cha != out_cha or stride > 1:
            self.conv_skip = MaskedConv1d('A' if first else 'B', inp_cha,  out_cha, 3, stride, 1, bias = False)
        else:
            self.conv_skip = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x) + self.conv_skip(residual)))
        return x

class ResNetBlockUnMasked(nn.Module):
    
    def __init__(self, inp_cha, out_cha, stride = 1, batch_norm = False):
        super().__init__()
        self.conv1 = nn.Conv1d(inp_cha, out_cha, 3, stride, 1)
        self.conv2 = nn.Conv1d(out_cha, out_cha, 3, 1, 1)
        #in_channels, out_channels, kernel_size, stride=1, padding=0

        self.act1 = nn.PReLU(out_cha, init = 0.2)
        self.act2 = nn.PReLU(out_cha, init = 0.2)

        if batch_norm:
            self.bn1 = nn.BatchNorm1d(out_cha)
            self.bn2 = nn.BatchNorm1d(out_cha)
        else:
            self.bn1 = nn.Identity()
            self.bn2 = nn.Identity()

        # If dimensions change, transform shortcut with a conv layer
        if inp_cha != out_cha or stride > 1:
            self.conv_skip = nn.Conv1d(inp_cha,  out_cha, 3, stride, 1, bias=False)
        else:
            self.conv_skip = nn.Identity()

    def forward(self, x):
        residual = x
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x) + self.conv_skip(residual)))
        return x

class AffineLayer(nn.Module):
    
    def __init__(self, COND_INPUTS, stride, h_cha = 96):
        # COND_INPUTS = COND_INPUTS + obs_dim = 1 + obs_dim = 4 by default (w/o CO2)
        super().__init__()
        self.feature_net = nn.Sequential(ResNetBlockUnMasked(COND_INPUTS, h_cha), ResNetBlockUnMasked(h_cha, COND_INPUTS))
        self.first_block = ResNetBlock(1, h_cha, first = True)
        self.second_block = nn.Sequential(ResNetBlock(h_cha + COND_INPUTS, h_cha, first = False), MaskedConv1d('B', h_cha,  2, 3, stride, 1, bias = False))
        
        self.unpack = True if COND_INPUTS > 1 else False

    def forward(self, x, COND_INPUTS): # x.shape == (batch_size, 1, n * state_dim)
        if self.unpack:
            COND_INPUTS = torch.cat([*COND_INPUTS], 1) # (batch_size, obs_dim + 1, n * state_dim)
        COND_INPUTS = self.feature_net(COND_INPUTS) # (batch_size, obs_dim + 1, n * state_dim)
        first_block = self.first_block(x) # (batch_size, h_cha, n * state_dim)
        feature_vec = torch.cat([first_block, COND_INPUTS], 1) # (batch_size, h_cha + obs_dim + 1, n * state_dim)
        output = self.second_block(feature_vec) # (batch_size, 2, n * state_dim)
        mu, sigma = torch.chunk(output, 2, 1) # (batch_size, 1, n * state_dim)
        sigma = LowerBound.apply(sigma, 1e-8)
        x = mu + sigma * x # (batch_size, 1, n * state_dim)
        return x, -torch.log(sigma) # each of shape (batch_size, 1, n * state_dim)

class PermutationLayer(nn.Module):
    
    def __init__(self, STATE_DIM, REVERSE = False):
        super().__init__()
        self.state_dim = STATE_DIM
        self.index_1 = torch.randperm(STATE_DIM)
        self.reverse = REVERSE

    def forward(self, x):
        B, S, L = x.shape # (batch_size, 1, state_dim * n)
        x_reshape = x.reshape(B, S, -1, self.state_dim) # (batch_size, 1, n, state_dim)
        if self.reverse:
            x_perm = x_reshape.flip(-2)[:, :, :, self.index_1]
        else:
            x_perm = x_reshape[:, :, :, self.index_1]
        x = x_perm.reshape(B, S, L)
        return x

class SoftplusLayer(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()
    
    def forward(self, x):
        # in.shape == out.shape == (batch_size, 1, n * state_dim)
        y = self.softplus(x)
        return y, -torch.log(-torch.expm1(-y))

class BatchRenormLayer(nn.Module):
    
    def __init__(self, num_inputs, momentum = 1e-2, eps = 1e-5, affine = True, init_r_max = 1., max_r_max = 3., r_max_step_size = 1e-4, init_d_max = 0, max_d_max = 5., d_max_step_size = 2.5e-4, batch_renorm_warmup_iter = 5000):
        super(BatchRenormLayer, self).__init__()

        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.rand(num_inputs)) if affine else torch.zeros(num_inputs)
        self.beta = nn.Parameter(torch.rand(num_inputs)) if affine else torch.zeros(num_inputs)

        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_std', torch.ones(num_inputs))

        self.init_r_max = init_r_max
        self.max_r_max = max_r_max
        self.init_d_max = init_d_max
        self.max_d_max = max_d_max
        self.r_max_step_size = r_max_step_size
        self.d_max_step_size = d_max_step_size        
        self.batch_renorm_warmup_iter = 10000
        self.training_iter = 0

    def get_r_max(self, training_iter, batch_renorm_warmup_iter, init_r_max, max_r_max, r_max_step_size):
        if training_iter < batch_renorm_warmup_iter:
            return init_r_max
        else:
            return torch.tensor((init_r_max + (training_iter - batch_renorm_warmup_iter) * r_max_step_size)).clamp_(init_r_max, max_r_max)

    def get_d_max(self, training_iter, batch_renorm_warmup_iter, init_d_max, max_d_max, d_max_step_size):
        if training_iter < batch_renorm_warmup_iter:
            return init_d_max
        else:
            return torch.tensor((init_d_max + (training_iter - batch_renorm_warmup_iter) * d_max_step_size)).clamp_(init_d_max, max_d_max)

    def forward(self, inputs):

        x = inputs.squeeze(1) # (batch_size, n * state_dim)

        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_std = x.std(0, unbiased = False) + self.eps

            self.r_max = self.get_r_max(self.training_iter, self.batch_renorm_warmup_iter, self.init_r_max, self.max_r_max, self.r_max_step_size)
            self.d_max = self.get_d_max(self.training_iter, self.batch_renorm_warmup_iter, self.init_d_max, self.max_d_max, self.d_max_step_size)

            print('r_max = ', self.r_max)
            print('d_max = ', self.d_max)            

            self.r = (self.batch_std.detach() / self.running_std).clamp_(1 / self.r_max, self.r_max)
            self.d = ((self.batch_mean.detach() - self.running_mean) / self.running_std).clamp_(-self.d_max, self.d_max)

            print('r = ', self.r)
            print('d = ', self.d)  

            x_hat = self.r * (x - self.batch_mean) / self.batch_std + self.d

            std = self.batch_std            

            self.running_mean += self.momentum * (self.batch_mean.detach() - self.running_mean)
            self.running_std += self.momentum * (self.batch_std.detach() - self.running_std)

            self.training_iter += 1
        else:
            # mean.shape == std.shape == (n * state_dim, )
            x_hat = (x - self.running_mean) / self.running_std # (batch_size, n * state_dim)

            std = self.running_std

        y = torch.exp(self.log_gamma) * x_hat + self.beta # (batch_size, n * state_dim)
        ildj = -self.log_gamma + torch.log(std) # (n * state_dim, )

        # y.shape == (batch_size, 1, n * state_dim), ildj.shape == (1, 1, n * state_dim)
        return y[:, None, :], ildj[None, None, :]
    
class SDEFlow(nn.Module):

    def __init__(self, DEVICE, OBS_MODEL, STATE_DIM, T, DT, N,
                 I_S_TENSOR = None, I_D_TENSOR = None, COND_INPUTS = 1, NUM_LAYERS = 5, POSITIVE = True,
                 REVERSE = False, BASE_STATE = False):
        super().__init__()
        self.device = DEVICE
        self.obs_model = OBS_MODEL
        self.state_dim = STATE_DIM
        self.t = T
        self.dt = DT
        self.n = N

        self.base_state = BASE_STATE
        if self.base_state:
            base_loc_SOC, base_loc_DOC, base_loc_MBC = torch.split(nn.Parameter(torch.zeros(1, self.state_dim)), 1, -1)
            base_scale_SOC, base_scale_DOC, base_scale_MBC = torch.split(nn.Parameter(torch.ones(1, self.state_dim)), 1, -1)
            base_loc = torch.cat((base_loc_SOC.expand([1, self.n]), base_loc_DOC.expand([1, self.n]), base_loc_MBC.expand([1, self.n])), 1)
            base_scale = torch.cat((base_scale_SOC.expand([1, self.n]), base_scale_DOC.expand([1, self.n]), base_scale_MBC.expand([1, self.n])), 1)
            self.base_dist = D.normal.Normal(loc = base_loc, scale = base_scale)                
        else:
            self.base_dist = D.normal.Normal(loc = 0., scale = 1.)

        self.cond_inputs = COND_INPUTS  
        if self.cond_inputs == 3:
            self.i_tensor = torch.stack((I_S_TENSOR.reshape(-1), I_D_TENSOR.reshape(-1)))[None, :, :].repeat_interleave(3, -1)

        self.num_layers = NUM_LAYERS
        self.reverse = REVERSE

        self.affine = nn.ModuleList([AffineLayer(COND_INPUTS + self.obs_model.obs_dim, 1) for _ in range(NUM_LAYERS)])
        self.permutation = [PermutationLayer(STATE_DIM, REVERSE = self.reverse) for _ in range(NUM_LAYERS)]
        self.batch_renorm = nn.ModuleList([BatchRenormLayer(STATE_DIM * N) for _ in range(NUM_LAYERS - 1)])
        self.positive = POSITIVE
        if self.positive:
            self.SP = SoftplusLayer()
        
    def forward(self, BATCH_SIZE, *args, **kwargs):
        if self.base_state:
            eps = self.base_dist.rsample([BATCH_SIZE]).to(self.device)
        else:
            eps = self.base_dist.rsample([BATCH_SIZE, 1, self.state_dim * self.n]).to(self.device)
        log_prob = self.base_dist.log_prob(eps).sum(-1) # (batch_size, 1)
        
        # NOTE: This currently assumes a regular time gap between observations!
        steps_bw_obs = self.obs_model.idx[1] - self.obs_model.idx[0]
        reps = torch.ones(len(self.obs_model.idx), dtype=torch.long).to(self.device) * self.state_dim
        reps[1:] *= steps_bw_obs
        obs_tile = self.obs_model.mu[None, :, :].repeat_interleave(reps, -1).repeat( \
            BATCH_SIZE, 1, 1).to(self.device) # (batch_size, obs_dim, state_dim * n)
        times = torch.arange(0, self.t + self.dt, self.dt, device = eps.device)[None, None, :].repeat( \
            BATCH_SIZE, self.state_dim, 1).transpose(-2, -1).reshape(BATCH_SIZE, 1, -1).to(self.device)
        
        if self.cond_inputs == 3:
            i_tensor = self.i_tensor.repeat(BATCH_SIZE, 1, 1).to(self.device)
            features = (obs_tile, times, i_tensor)
        else:
            features = (obs_tile, times)

        ildjs = []
        
        for i in range(self.num_layers):
            eps, cl_ildj = self.affine[i](self.permutation[i](eps), features) # (batch_size, 1, n * state_dim)
            if i < (self.num_layers - 1):
                eps, bn_ildj = self.batch_renorm[i](eps) # (batch_size, 1, n * state_dim), (1, 1, n * state_dim)
                ildjs.append(bn_ildj)
            ildjs.append(cl_ildj)
                
        if self.positive:
            eps, sp_ildj = self.SP(eps) # (batch_size, 1, n * state_dim)
            ildjs.append(sp_ildj)
        
        eps = eps.reshape(BATCH_SIZE, -1, self.state_dim) + 1e-6 # (batch_size, n, state_dim)
        for ildj in ildjs:
            log_prob += ildj.sum(-1) # (batch_size, 1)
    
        #return eps.reshape(BATCH_SIZE, self.state_dim, -1).permute(0, 2, 1) + 1e-6, log_prob
        return eps, log_prob # (batch_size, n, state_dim), (batch_size, 1)

###################################################
##OBSERVATION MODEL RELATED CLASSES AND FUNCTIONS##
###################################################

class ObsModel(nn.Module):

    def __init__(self, DEVICE, TIMES, DT, MU, SCALE):
        super().__init__()
        self.times = TIMES # (n_obs, )
        self.dt = DT
        self.idx = self.get_idx(TIMES, DT)        
        self.mu = torch.Tensor(MU).to(DEVICE) # (obs_dim, n_obs)
        self.scale = torch.Tensor(SCALE).to(DEVICE) # (1, obs_dim)
        self.obs_dim = self.mu.shape[0]
        
    def forward(self, x):
        obs_ll = D.normal.Normal(self.mu.permute(1, 0), self.scale).log_prob(x[:, self.idx, :])
        return torch.sum(obs_ll, [-1, -2]).mean()

    def get_idx(self, TIMES, DT):
        return list((TIMES / DT).astype(int))
    
    def plt_dat(self):
        return self.mu, self.times

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
