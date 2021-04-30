import torch
from torch import nn
import torch.distributions as d
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
    obs_error_sd = torch.mean(obs_means_T, 1) * obs_error_scale
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

class CouplingLayer(nn.Module):
    
    def __init__(self, cond_inputs, stride, h_cha = 96):
        # cond_inputs = cond_inputs + STATE_DIM = 1 + STATE_DIM = 4 by default
        super().__init__()
        self.feature_net = nn.Sequential(ResNetBlockUnMasked(cond_inputs, h_cha),
        #self.feature_net = nn.Sequential(ResNetBlockUnMasked(cond_inputs, h_cha),
                                         ResNetBlockUnMasked(h_cha, cond_inputs))
        self.first_block = ResNetBlock(1, h_cha, first = True)
        self.second_block = nn.Sequential(ResNetBlock(h_cha + cond_inputs, h_cha, first = False),
                                          MaskedConv1d('B', h_cha,  2, 3, stride, 1, bias = False))
        
        self.unpack = True if cond_inputs > 1 else False

    def forward(self, x, cond_inputs):
        if self.unpack:
            cond_inputs = torch.cat([*cond_inputs], 1) # (batch_size, obs_dim + 1, n * state_dim)
        #print(cond_inputs[0, :, 0], cond_inputs[0, :, 60], cond_inputs[0, :, 65])
        cond_inputs = self.feature_net(cond_inputs)
        first_block = self.first_block(x)
        #print(first_block.shape, cond_inputs.shape)
        feature_vec = torch.cat([first_block, cond_inputs], 1)
        output = self.second_block(feature_vec)
        mu, sigma = torch.chunk(output, 2, 1)
        #print('mu and sigma shapes:', mu.shape, sigma.shape)
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

    def __init__(self, DEVICE, OBS_MODEL, STATE_DIM, T, DT, N, I_S_tensor, I_D_tensor, cond_inputs = 1, num_layers = 5):
        super().__init__()
        self.device = DEVICE
        self.obs_model = OBS_MODEL
        self.state_dim = STATE_DIM
        self.t = T
        self.dt = DT
        self.n = N
        self.i_tensor = torch.stack((I_S_tensor.reshape(-1), I_D_tensor.reshape(-1)))[None, :, 1:].repeat_interleave(3, -1)

        self.base_dist = d.normal.Normal(loc = 0., scale = 1.)
        self.cond_inputs = cond_inputs        
        self.num_layers = num_layers

        self.coupling = nn.ModuleList([CouplingLayer(cond_inputs + self.obs_model.obs_dim, 1) for _ in range(num_layers)])
        self.permutation = [PermutationLayer(STATE_DIM) for _ in range(num_layers)]
        self.batch_norm = nn.ModuleList([BatchNormLayer(STATE_DIM * N) for _ in range(num_layers - 1)])
        self.SP = SoftplusLayer()
        
    def forward(self, BATCH_SIZE, *args, **kwargs):
        eps = self.base_dist.sample([BATCH_SIZE, 1, self.state_dim * self.n]).to(self.device)
        log_prob = self.base_dist.log_prob(eps).sum(-1)
        
        #obs_tile = self.obs_model.mu[None, :, 1:, None].repeat(self.batch_size, self.state_dim, 1, 50).reshape(self.batch_size, self.state_dim, -1)
        #times = torch.arange(self.dt, self.t + self.dt, self.dt, device = eps.device)[(None,) * 2].repeat(self.batch_size, self.state_dim, 1).transpose(-2, -1).reshape(self.batch_size, 1, -1)
        #obs_dim = self.state_dim + 1
        #obs_tile = self.obs_model.mu[None, :, 1:, None].repeat(self.batch_size, self.state_dim, 1, 50).reshape(self.batch_size, self.obs_model.obs_dim, -1)
        #obs_tile = self.obs_model.mu[None, :, 1:, None].repeat( \
        #    self.batch_size, 1, self.state_dim, steps_bw_obs).reshape(self.batch_size, self.obs_model.obs_dim, -1)

        # NOTE: This currently assumes a regular time gap between observations!
        steps_bw_obs = self.obs_model.idx[1] - self.obs_model.idx[0]
        #print(steps_bw_obs)
        obs_tile = self.obs_model.mu[None, :, 1:].repeat_interleave(self.state_dim * steps_bw_obs, -1).repeat( \
            BATCH_SIZE, 1, 1) # (batch_size, obs_dim, state_dim * num_steps)
        times = torch.arange(self.dt, self.t + self.dt, self.dt, device = eps.device)[(None,) * 2].repeat( \
            BATCH_SIZE, self.state_dim, 1).transpose(-2, -1).reshape(BATCH_SIZE, 1, -1)
        i_tensor = self.i_tensor.repeat(BATCH_SIZE, 1, 1)
        #print(obs_tile)

        ildjs = []
        
        for i in range(self.num_layers):
            eps, cl_ildj = self.coupling[i](self.permutation[i](eps), (obs_tile, times, i_tensor))
            if i < (self.num_layers - 1):
                eps, bn_ildj = self.batch_norm[i](eps)
                ildjs.append(bn_ildj)
            ildjs.append(cl_ildj)
                
        eps, sp_ildj = self.SP(eps)
        ildjs.append(sp_ildj)
        
        for ildj in ildjs:
            log_prob += ildj.sum(-1)
    
        # x.shape == (batch_size, N, state_dim), log_prob.shape == (batch_size, 1)
        return eps.reshape(BATCH_SIZE, self.state_dim, -1).permute(0, 2, 1) + 1e-6, log_prob

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
        self.mu = torch.Tensor(MU).to(DEVICE) # (obs_dim, num_obs)
        self.scale = SCALE
        self.obs_dim = self.mu.shape[0]
        
    def forward(self, x, theta):
        obs_ll = d.normal.Normal(self.mu.permute(1, 0), self.scale).log_prob(x[:, self.idx, :])
        return torch.sum(obs_ll, [-1, -2]).mean()

    def get_idx(self, TIMES, DT):
        return list((TIMES / DT).astype(int))
    
    def plt_dat(self):
        return self.mu, self.times

class ObsModelCO2(ObsModel):

    '''
    ObsModelCO2 is a derived class of ObsModel and needs to inherit ObsModel's classes. CO2 mu should already be a part of ObsModel.mu following read-in from data.
    '''

    def __init__(self, DEVICE, TIMES, DT, MU, SCALE, GET_CO2, T_SPAN_TENSOR, TEMP_GEN, TEMP_REF):
        super().__init__(DEVICE, TIMES, DT, MU, SCALE)
        self.get_CO2 = GET_CO2
        self.t_span_tensor = T_SPAN_TENSOR
        self.temp_gen = TEMP_GEN
        self.temp_ref = TEMP_REF

    def forward(self, x, theta):
        CO2 = self.get_CO2(x, self.t_span_tensor, theta, self.temp_gen, self.temp_ref)
        x_with_CO2 = torch.cat((x[:, self.idx, :], CO2[:, self.idx, :]), dim = -1)
        #print(self.mu.permute(1, 0), x_with_CO2)
        obs_ll = d.normal.Normal(self.mu.permute(1, 0), self.scale).log_prob(x_with_CO2)
        return torch.sum(obs_ll, [-1, -2]).mean()

###################################################
## PARAMETER MODEL RELATED CLASSES AND FUNCTIONS ##
###################################################

class MeanField(nn.Module):
    def __init__(self, init_params):
        super().__init__()

        # use param dict to intialise the means for the mean-field approximations
        means = []
        keys = []
        for key, value in init_params.items():
            keys += [key]
            means += [value]
        self.means = nn.Parameter(torch.Tensor(means))
        # save keys for forward output
        self.keys = keys

        # initial std == 1
        self.std = nn.Parameter(torch.ones(self.means.shape) * 0.1)

    def forward(self, n=30, return_sample_mean=True):
        # define q(theta)
        q_dist = d.normal.Normal(self.means, LowerBound.apply(self.std, 1e-6))
        # sample theta ~ q(theta)
        print(samples.shape)
        samples = q_dist.rsample([n]) # (num_samples, num_params)
        # evaluate log prob of theta samples
        log_probs = torch.sum(q_dist.log_prob(samples), -1) # shape of n
        # return samples in same diitionary format
        dict_out = {} # define dicitonary with n samples for each parameter
        for key, sample in zip(self.keys, torch.split(samples, 1, -1),):
            if return_sample_mean:
                dict_out[f"{key}"] = sample.mean() 
            else:
                dict_out[f"{key}"] = sample # each sample is of shape [n, 1]
        # returns samples in dicitionary and tensor format
        return dict_out, samples, log_probs

###################################################
##ELBO AND TRAINING RELATED CLASSES AND FUNCTIONS##
###################################################

# def neg_log_lik(C_path, T_span_tensor, dt, I_S_tensor, I_D_tensor, drift_diffusion, params_dict, temp_ref):
#     drift, diffusion_sqrt = drift_diffusion(C_path[:, :-1, :], T_span_tensor[:, :-1, :], I_S_tensor[:, :-1, :], I_D_tensor[:, :-1, :], params_dict, temp_ref)
#     #print('\n drift =', drift)
#     #print('\n diffusion_sqrt =', diffusion_sqrt)
#     #euler_maruyama_sample = d.multivariate_normal.MultivariateNormal(loc = C_path[:, :-1, :] + drift * dt, scale_tril = diffusion_sqrt * math.sqrt(dt)) This line no longer applies because of addition of CO2 as a 'state'.
#     drift_means_with_CO2 = torch.cat((C_path[:, :-1, :-1] + drift[:, :, :-1] * dt, drift[:, :, -1].unsqueeze(2)), 2) #Separate explicit algebraic variable CO2 mean from integration process.
#     euler_maruyama_sample = d.multivariate_normal.MultivariateNormal(loc = drift_means_with_CO2, scale_tril = diffusion_sqrt * math.sqrt(dt))
#     return -euler_maruyama_sample.log_prob(C_path[:, 1:, :]).sum(-1)
#
# def train(niter, pretrain_iter, BATCH_SIZE, T_span_tensor, I_S_tensor, I_D_tensor, drift_diffusion, params_dict, analytical_steady_state_init):
#     if pretrain_iter >= niter:
#         raise Exception("pretrain_inter must be < niter.")
#     best_loss_norm = 1e10
#     best_loss_ELBO = 1e20
#     norm_losses = [best_loss_norm] * 10
#     ELBO_losses = [best_loss_ELBO] * 10
#     C0 = analytical_steady_state_init(I_S_tensor[0, 0, 0].item(), I_D_tensor[0, 0, 0].item(), params_dict) #Calculate deterministic initial conditions.
#     C0 = C0[(None,) * 2].repeat(BATCH_SIZE, 1, 1).to(device) #Assign initial conditions to C_path.
#     with tqdm(total = niter, desc = f'Train Diffusion', position = -1) as t:
#         for iter in range(niter):
#             net.train()
#             optimizer.zero_grad()
#             C_path, log_prob = net(BATCH_SIZE, obs_model) #Obtain paths with solutions at times after t0.
#             C_path = torch.cat([C0, C_path], 1) #Append deterministic CON initial conditions conditional on parameter values to C path. 
#             if iter <= pretrain_iter:
#                 l1_norm_element = C_path - torch.mean(obs_model.mu, -1)
#                 l1_norm = torch.sum(torch.abs(l1_norm_element)).mean()
#                 best_loss_norm = l1_norm if l1_norm < best_loss_norm else best_loss_norm
#                 l1_norm.backward()
#                 norm_losses.append(l1_norm.item())
#                 #l2_norm_element = C_path - torch.mean(obs_model.mu, -1)
#                 #l2_norm = torch.sqrt(torch.sum(torch.square(l2_norm_element))).mean()
#                 #best_loss_norm = l2_norm if l2_norm < best_loss_norm else best_loss_norm
#                 #l2_norm.backward()
#                 #norm_losses.append(l2_norm.item())
#                 if len(norm_losses) > 10:
#                     norm_losses.pop(0)
#                 if iter % 10 == 0:
#                     print(f"Moving average norm loss at {iter} iterations is: {sum(norm_losses) / len(norm_losses)}. Best norm loss value is: {best_loss_norm}.")
#                     print('\nC_path mean =', C_path.mean(-2))
#                     print('\nC_path =', C_path)
#             else:
#                 log_lik = neg_log_lik(C_path, T_span_tensor.to(device), dt, I_S_tensor.to(device), I_D_tensor.to(device), drift_diffusion, params_dict, temp_ref)
#                 ELBO = log_prob.mean() + log_lik.mean() - obs_model(C_path)
#                 best_loss_ELBO = ELBO if ELBO < best_loss_ELBO else best_loss_ELBO
#                 ELBO.backward()
#                 ELBO_losses.append(ELBO.item())
#                 if len(ELBO_losses) > 10:
#                     ELBO_losses.pop(0)
#                 if iter % 10 == 0:
#                     print(f"Moving average ELBO loss at {iter} iterations is: {sum(ELBO_losses) / len(ELBO_losses)}. Best ELBO loss value is: {best_loss_ELBO}.")
#                     print('\nC_path mean =', C_path.mean(-2))
#                     print('\n C_path =', C_path)
#             torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
#             optimizer.step()
#             if iter % 100000 == 0 and iter > 0:
#                 optimizer.param_groups[0]['lr'] *= 0.1
#             t.update()
