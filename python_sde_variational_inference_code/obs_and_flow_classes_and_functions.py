import torch
from torch import nn
import torch.distributions as D
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
from tqdm import tqdm
import random
from torch.autograd import Function
# from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import sys
from pathlib import Path
import shutil
import pandas as pd

###########################################
##SYNTHETIC OBSERVATION READ-IN FUNCTIONS##
###########################################

def csv_to_obs_df(df_csv_string, dim, T, obs_error_scale_factor):
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
    obs_error_sd = torch.mean(obs_means_T, 1) * obs_error_scale_factor
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
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kW // 2 + 1 * (mask_type == 'B'):] = 0

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
        self.conv1 = nn.Conv1d(inp_cha,  out_cha, 3, stride, 1)
        self.conv2 = nn.Conv1d(out_cha,  out_cha, 3, 1, 1)

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

class CouplingLayer(nn.Module):
    
    def __init__(self, cond_inputs, stride, h_cha = 96):
        super().__init__()
        self.first_block = ResNetBlock(1, h_cha, first = True)
        self.second_block = nn.Sequential(ResNetBlock(h_cha + cond_inputs, h_cha, first = False),
                                          MaskedConv1d('B', h_cha,  2, 3, stride, 1, bias = False))

        self.feature_net = nn.Sequential(ResNetBlockUnMasked(cond_inputs, h_cha),
                                          ResNetBlockUnMasked(h_cha, cond_inputs))
        
        self.unpack = True if cond_inputs > 1 else False

    def forward(self, x, cond_inputs):
        if self.unpack:
            cond_inputs = torch.cat([*cond_inputs], 1)
        cond_inputs = self.feature_net(cond_inputs)
        first_block = self.first_block(x)
        feature_vec = torch.cat([first_block, cond_inputs], 1)
        output = self.second_block(feature_vec)
        mu, sigma = torch.chunk(output, 2, 1)
        sigma = LowerBound.apply(sigma, 1e-6)
        x = mu + sigma * x
        return x, -torch.log(sigma)

class PermutationLayer(nn.Module):
    
    def __init__(self, STATE_DIM):
        super().__init__()
        self.state_dim = STATE_DIM
        self.index_1 = torch.randperm(STATE_DIM)

    def forward(self, x):
        B, S, L = x.shape
        x_reshape = x.reshape(B, S, -1, self.state_dim)
        x_perm = x_reshape[:, :, :, self.index_1]
        x = x_perm.reshape(B, S, L)
        return x

class SoftplusLayer(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()
    
    def forward(self, x):
        y = self.softplus(x)
        return y, -torch.log(-torch.expm1(-y))

class BatchNormLayer(nn.Module):
    
    def __init__(self, num_inputs, momentum = 0.0, eps = 1e-5):
        super(BatchNormLayer, self).__init__()

        self.log_gamma = nn.Parameter(torch.rand(num_inputs))
        self.beta = nn.Parameter(torch.rand(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, inputs):
        inputs = inputs.squeeze(1)
        if self.training:
            self.batch_mean = inputs.mean(0)
            self.batch_var = (inputs - self.batch_mean).pow(2).mean(0) + self.eps

            self.running_mean.mul_(self.momentum)
            self.running_var.mul_(self.momentum)

            self.running_mean.add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.add_(self.batch_var.data * (1 - self.momentum))

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (inputs - mean) / var.sqrt()
        y = torch.exp(self.log_gamma) * x_hat + self.beta
        ildj = -self.log_gamma + 0.5 * torch.log(var)
        return y[:, None, :], ildj[None, None, :]
    
class SDEFlow(nn.Module):

    def __init__(self, DEVICE, BATCH_SIZE, OBS_MODEL, STATE_DIM, T, DT, N, cond_inputs = 1, num_layers = 5):
        super().__init__()
        self.device = DEVICE
        self.batch_size = BATCH_SIZE
        self.obs_model = OBS_MODEL
        self.state_dim = STATE_DIM
        self.t = T
        self.dt = DT
        self.n = N

        self.base_dist = D.normal.Normal(loc = 0., scale = 1.)
        self.cond_inputs = cond_inputs        
        self.num_layers = num_layers

        self.coupling = nn.ModuleList([CouplingLayer(cond_inputs + STATE_DIM, 1) for _ in range(num_layers)])
        self.permutation = [PermutationLayer(STATE_DIM) for _ in range(num_layers)]
        self.batch_norm = nn.ModuleList([BatchNormLayer(STATE_DIM * N) for _ in range(num_layers - 1)])
        self.SP = SoftplusLayer()
        
    def forward(self, *args, **kwargs):
        eps = self.base_dist.sample([self.batch_size, 1, self.state_dim * self.n]).to(self.device)
        log_prob = self.base_dist.log_prob(eps).sum(-1)
        
        obs_tile = self.obs_model.mu[None, :, 1:, None].repeat(self.batch_size, self.state_dim, 1, 50).reshape(self.batch_size, self.state_dim, -1)
        times = torch.arange(self.dt, self.t + self.dt, self.dt, device = eps.device)[(None,) * 2].repeat(self.batch_size, self.state_dim, 1).transpose(-2, -1).reshape(self.batch_size, 1, -1)
        ildjs = []
        
        for i in range(self.num_layers):
            eps, cl_ildj = self.coupling[i](self.permutation[i](eps), (obs_tile, times))
            if i < (self.num_layers - 1):
                eps, bn_ildj = self.batch_norm[i](eps)
                ildjs.append(bn_ildj)
            ildjs.append(cl_ildj)
                
        eps, sp_ildj = self.SP(eps)
        ildjs.append(sp_ildj)
        
        for ildj in ildjs:
            log_prob += ildj.sum(-1)
    
        return eps.reshape(self.batch_size, self.state_dim, -1).permute(0, 2, 1) + 1e-6, log_prob

###################################################
##OBSERVATION MODEL RELATED CLASSES AND FUNCTIONS##
###################################################

class ObsModel(nn.Module):

    def __init__(self, DEVICE, TIMES, DT, MU, SCALE):
        super().__init__()
        self.device = DEVICE
        self.times = TIMES
        self.dt = DT
        self.idx = self.get_idx(TIMES, DT)        
        self.mu = torch.Tensor(MU).to(DEVICE)
        self.scale = SCALE
        
    def forward(self, x):
        obs_ll = D.normal.Normal(self.mu.permute(1, 0), self.scale).log_prob(x[:, self.idx, :])
        return torch.sum(obs_ll, [-1, -2]).mean()

    def get_idx(self, TIMES, DT):
        return list((TIMES / DT).astype(int))
    
    def plt_dat(self):
        return self.mu, self.times

class ObsModelCO2(ObsModel):

    '''
    ObsModelCO2 is a derived class of ObsModel and needs to inherit ObsModel's classes. CO2 mu should already be a part of ObsModel.mu following read-in from data.
    '''

    def __init__(self, DEVICE, TIMES, DT, MU, SCALE, GET_CO2):
        super().__init__(DEVICE, TIMES, DT, MU, SCALE)
        self.get_CO2 = GET_CO2

    def forward(self, x, T_SPAN_TENSOR, PARAMS_DICT, TEMP_GEN, TEMP_REF):
        CO2 = self.get_CO2(x[:, self.idx, :], T_SPAN_TENSOR[:, self.idx, :], PARAMS_DICT, TEMP_GEN, TEMP_REF)
        x_with_CO2 = torch.cat((x[:, self.idx, :], CO2), dim = -1)
        obs_ll = D.normal.Normal(self.mu.permute(1, 0), self.scale).log_prob(x_with_CO2)
        return torch.sum(obs_ll, [-1, -2]).mean()
