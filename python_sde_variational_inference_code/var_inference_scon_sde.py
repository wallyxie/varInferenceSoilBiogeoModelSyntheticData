import torch
from torch import nn
import torch.distributions as d
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
import random
from torch.autograd import Function
import argparse
import os
import sys
from pathlib import Path
import shutil
import pandas as pd

torch.manual_seed(0)
STATE_DIM = 3
CUDA_ID = 1
dt = .2
T = 1000 #Run simulation for 1000 hours.
N = int(T / dt) 
T_span = np.linspace(0, T, N + 1)
T_span_tensor = torch.reshape(torch.Tensor(T_span), [1, N + 1, 1]) #T_span needs to be converted to tensor object. Additionally, facilitates conversion of I_S and I_D to tensor objects.

device = torch.device("".join(["cuda:",f'{CUDA_ID}']) if torch.cuda.is_available() else "cpu")
LR = 1e-3
niter = 100000

obs_df_full = pd.read_csv('CON_synthetic_sol_df.csv') #Must be link to raw Github output if in Colab.
obs_df = obs_df_full[obs_df_full['hour'] <= T] #Test with just first 1,000 hours of data.
obs_times = np.array(obs_df['hour'])

obs_means = torch.Tensor(np.array(obs_df.drop(columns = 'hour')))

obs_SOC, obs_DOC, obs_MBC = [obs_means[:, i : i + 1] for i in range(3)]
obs_means_re = torch.reshape(torch.cat([obs_SOC, obs_DOC, obs_MBC], 0), [3, len(obs_SOC)])
obs_means_T = obs_means.T #Transposing obs_means data also does the above transformation in an easier manner.

I_S_tensor = 0.001 + 0.0005 * torch.sin((2 * math.pi / (24 * 365)) * T_span_tensor) #Exogenous SOC input function
I_D_tensor = 0.0001 + 0.00005 * torch.sin((2 * math.pi / (24 * 365)) * T_span_tensor) #Exogenous DOC input function

temp_ref = 283

u_M = 0.002
a_SD = 0.33
a_DS = 0.33
a_M = 0.33
a_MSC = 0.5
k_S_ref = 0.000025
k_D_ref = 0.005
k_M_ref = 0.0002
Ea_S = 75
Ea_D = 50
Ea_M = 50

scon_params_dict = {'u_M': u_M, 'a_SD': a_SD, 'a_DS': a_DS, 'a_M': a_M, 'a_MSC': a_MSC, 'k_S_ref': k_S_ref, 'k_D_ref': k_D_ref, 'k_M_ref': k_M_ref, 'Ea_S': Ea_S, 'Ea_D': Ea_D, 'Ea_M': Ea_M}

############################################################
##SOIL BIOGEOCHEMICAL MODEL TEMPERATURE RESPONSE FUNCTIONS##
############################################################

def temp_gen(t, temp_ref):
    temp = temp_ref + t / (20 * 24 * 365) + 10 * torch.sin((2 * np.pi / 24) * t) + 10 * torch.sin((2 * math.pi / (24 * 365)) * t)
    return temp

def arrhenius_temp_dep(parameter, temp, Ea, temp_ref):
    '''
    For a parameter with Arrhenius temperature dependence, returns the transformed parameter value.
    0.008314 is the gas constant. Temperatures are in K.
    '''
    decayed_parameter = parameter * torch.exp(-Ea / 0.008314 * (1 / temp - 1 / temp_ref))
    return decayed_parameter

def linear_temp_dep(parameter, temp, Q, temp_ref):
    '''
    For a parameter with linear temperature dependence, returns the transformed parameter value.
    Q is the slope of the temperature dependence and is a varying parameter.
    Temperatures are in K.
    '''
    modified_parameter = parameter - Q * (temp - temp_ref)
    return modified_parameter

##########################################################################
##DETERMINISTIC SOIL BIOGEOCHEMICAL MODEL INITIAL STEADY STATE ESTIMATES##
##########################################################################

#Analytical_steady_state_init_awb to be coded later.
def analytical_steady_state_init_con(SOC_input, DOC_input, scon_params_dict):
    '''
    Returns a vector of C pool values to initialize an SCON system corresponding to set of parameter values using the analytical steady state equations of the deterministic CON system.
    Vector elements are in order of S_0, D_0, and M_0.
    Expected scon_params_dict = {scon_params_dict = {'u_M': u_M, 'a_SD': a_SD, 'a_DS': a_DS, 'a_M': a_M, 'a_MSC': a_MSC, 'k_S_ref': k_S_ref, 'k_D_ref': k_D_ref, 'k_M_ref': k_M_ref, 'Ea_S': Ea_S, 'Ea_D': Ea_D, 'Ea_M': Ea_M}    
    '''
    D_0 = (DOC_input + SOC_input * scon_params_dict['a_SD']) / (scon_params_dict['u_M'] + scon_params_dict['k_D_ref'] + scon_params_dict['u_M'] * scon_params_dict['a_M'] * (scon_params_dict['a_MSC'] - 1 - scon_params_dict['a_MSC'] * scon_params_dict['a_SD']) - scon_params_dict['a_DS'] * scon_params_dict['k_D_ref'] * scon_params_dict['a_SD'])
    S_0 = (SOC_input + D_0 * (scon_params_dict['a_DS'] * scon_params_dict['k_D_ref'] + scon_params_dict['u_M'] * scon_params_dict['a_M'] * scon_params_dict['a_MSC'])) / scon_params_dict['k_S_ref']
    M_0 = scon_params_dict['u_M'] * D_0 / scon_params_dict['k_M_ref']
    C_0_vector = torch.as_tensor([S_0, D_0, M_0])
    return C_0_vector

####################################################
##STOCHASTIC DIFFERENTIAL EQUATION MODEL FUNCTIONS##
#################################################### 

def drift_diffusion_scon(C_vector, T_span_tensor, I_S_tensor, I_D_tensor, scon_params_dict, temp_ref):
    '''
    Returns SCON drift vectors and diffusion matrices.
    current_temp is output from temp_gen function. 
    Expected scon_params_dict = {scon_params_dict = {'u_M': u_M, 'a_SD': a_SD, 'a_DS': a_DS, 'a_M': a_M, 'a_MSC': a_MSC, 'k_S_ref': k_S_ref, 'k_D_ref': k_D_ref, 'k_M_ref': k_M_ref, 'Ea_S': Ea_S, 'Ea_D': Ea_D, 'Ea_M': Ea_M}
    '''
    STATE_DIM = 3 #SCON and CON will always have three state variables.
    SOC, DOC, MBC = [C_vector[:, :, l : l + 1] for l in range(STATE_DIM)] #Assign SOC, DOC, and MBC values.
    drift_vector = C_vector.clone() #Create tensor to assign drift.
    diffusion_matrix_sqrt = torch.zeros([drift_vector.size(0), drift_vector.size(1), STATE_DIM, STATE_DIM], device = drift_vector.device) #Create tensor to assign diffusion matrix elements.
    current_temp = temp_gen(T_span_tensor, temp_ref) #Obtain temperature function vector across span of times.
    #Decay parameters are forced by temperature changes.
    k_S = arrhenius_temp_dep(scon_params_dict['k_S_ref'], current_temp, scon_params_dict['Ea_S'], temp_ref) #Apply vectorized temperature-dependent transformation to k_S_ref.
    k_D = arrhenius_temp_dep(scon_params_dict['k_D_ref'], current_temp, scon_params_dict['Ea_D'], temp_ref) #Apply vectorized temperature-dependent transformation to k_D_ref.
    k_M = arrhenius_temp_dep(scon_params_dict['k_M_ref'], current_temp, scon_params_dict['Ea_M'], temp_ref) #Apply vectorized temperature-dependent transformation to k_M_ref.
    #Drift is calculated.
    drift_SOC = I_S_tensor + scon_params_dict['a_DS'] * k_D * DOC + scon_params_dict['a_M'] * scon_params_dict['a_MSC'] * k_M * MBC - k_S * SOC
    drift_DOC = I_D_tensor + scon_params_dict['a_SD'] * k_S * SOC + scon_params_dict['a_M'] * (1 - scon_params_dict['a_MSC']) * k_M * MBC - ((scon_params_dict['u_M']) + k_D) * DOC
    drift_MBC = scon_params_dict['u_M'] * DOC - k_M * MBC
    #Assign elements to drift vector.
    drift_vector[:, :, 0 : 1] = drift_SOC
    drift_vector[:, :, 1 : 2] = drift_DOC
    drift_vector[:, :, 2 : 3] = drift_MBC
    #Diffusion matrix is assigned with Cholesky factorization (torch.abs does the same thing as torch.cholesky(torch.mm(diffusion_matrix, diffusion_matrix.t))) to ensure positivite definiteness of diagonal diffusion matrix. LowerBound in case elements are too close to 0.
    diffusion_matrix_sqrt[:, :, 0 : 1, 0] = torch.sqrt(LowerBound.apply(torch.abs(drift_SOC), 1e-2)) #SOC diffusion standard deviation bound is higher because SOC is higher in path than the other state variables. 
    diffusion_matrix_sqrt[:, :, 1 : 2, 1] = torch.sqrt(LowerBound.apply(torch.abs(drift_DOC), 1e-6)) #DOC diffusion standard deviation is lowest.
    diffusion_matrix_sqrt[:, :, 2 : 3, 2] = torch.sqrt(LowerBound.apply(torch.abs(drift_MBC), 1e-4)) #MBC diffusion standard deviation
    return drift_vector, diffusion_matrix_sqrt

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
        self.mask[:, :, kW // 2 + 1*(mask_type == 'B'):] = 0

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
        x = mu + sigma*x
        return x, -torch.log(sigma)

class PermutationLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.index_1 = torch.randperm(STATE_DIM)

    def forward(self, x):
        B, S, L = x.shape
        x_reshape = x.reshape(B, S, -1, STATE_DIM)
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
            self.batch_var = (
                inputs - self.batch_mean).pow(2).mean(0) + self.eps

            self.running_mean.mul_(self.momentum)
            self.running_var.mul_(self.momentum)

            self.running_mean.add_(self.batch_mean.data *
                                   (1 - self.momentum))
            self.running_var.add_(self.batch_var.data *
                                  (1 - self.momentum))

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

    def __init__(self, cond_inputs = 1, num_layers = 5):
        super().__init__()
        
        self.coupling = nn.ModuleList([CouplingLayer(cond_inputs + STATE_DIM, 1) for _ in range(num_layers)])
        self.permutation = [PermutationLayer() for _ in range(num_layers)]
        self.batch_norm = nn.ModuleList([BatchNormLayer(STATE_DIM * N) for _ in range(num_layers-1)])
        self.SP = SoftplusLayer()
        
        self.base_dist = d.normal.Normal(loc = 0., scale = 1.)
        self.num_layers = num_layers
        
    def forward(self, batch_size, obs_model, *args, **kwargs):

        eps = self.base_dist.sample([batch_size, 1, STATE_DIM * N]).to(device)
        log_prob = self.base_dist.log_prob(eps).sum(-1)
        
        obs_tile = obs_model.mu[None, :, 1:, None].repeat(batch_size, STATE_DIM, 1, 50).reshape(batch_size, STATE_DIM, -1)
        times = torch.arange(dt, T + dt, dt, device = eps.device)[(None,) * 2].repeat(batch_size, STATE_DIM, 1).transpose(-2, -1).reshape(batch_size, 1, -1)
        
        ildjs = []
        
        for i in range(self.num_layers):
            eps, cl_ildj = self.coupling[i](self.permutation[i](eps), (obs_tile, times))
            if i < (self.num_layers-1):
                eps, bn_ildj = self.batch_norm[i](eps)
                ildjs.append(bn_ildj)
            ildjs.append(cl_ildj)
                
        eps, sp_ildj = self.SP(eps)
        ildjs.append(sp_ildj)
        
        for ildj in ildjs:
            log_prob += ildj.sum(-1)
    
        return eps.reshape(batch_size, STATE_DIM, -1).permute(0, 2, 1) + 1e-9, log_prob

class ObsModel(nn.Module):

    def __init__(self, times, mu, scale):
        super().__init__()

        self.idx = self._get_idx(times)
        self.times = times
        self.mu = torch.Tensor(mu).to(device)
        self.scale = scale
        
    def forward(self, x):
            obs_ll = d.normal.Normal(self.mu.permute(1, 0), self.scale).log_prob(x[:, self.idx, :])
            return torch.sum(obs_ll, [-1, -2]).mean()

    def _get_idx(self, times):
        return list((times / dt).astype(int))
    
    def plt_dat(self):
        return self.mu, self.times

def neg_log_lik_scon(C_vector, T_span_tensor, dt, I_S_tensor, I_D_tensor, drift_diffusion_scon, scon_params_dict, temp_ref):
    drift, diffusion_sqrt = drift_diffusion_scon(C_vector[:, :-1, :], T_span_tensor[:, :-1, :], I_S_tensor[:, :-1, :], I_D_tensor[:, :-1, :], scon_params_dict, temp_ref)
    euler_maruyama_sample = d.multivariate_normal.MultivariateNormal(loc = C_vector[:, :-1, :] + drift * dt, scale_tril = diffusion_sqrt * math.sqrt(dt))
    return -euler_maruyama_sample.log_prob(C_vector[:, 1:, :]).sum(-1)

obs_model = ObsModel(times = obs_times, mu = obs_means_T, scale = torch.tensor([1., 0.01, 0.1]).reshape([1, 3])) #Have different observation standard deviation for SOC, DOC, and MBC because of different scales of state variables.
net = SDEFlow().to(device)
optimizer = optim.Adam(net.parameters(), lr = LR)

def train_scon(niter, BATCH_SIZE, T_span_tensor, I_S_tensor, I_D_tensor, scon_params_dict):
    best_loss = 1e20
    losses = [1e20] * 100
    C0 = analytical_steady_state_init_con(I_S_tensor[0, 0, 0].item(), I_D_tensor[0, 0, 0].item(), scon_params_dict) #Calculate deterministic initial conditions.
    #print('\n C0 =', C0)
    C0 = C0[(None,) * 2].repeat(BATCH_SIZE, 1, 1).to(device) #Assign initial conditions to C_vector.
    with tqdm(total = niter, desc=f'Train Diffusion', position = -1) as t:
        for iter in range(niter):
            net.train()
            optimizer.zero_grad()
            C_vector, log_prob = net(BATCH_SIZE, obs_model) #Obtain paths with solutions at times after t0.
            C_vector = torch.cat([C0, C_vector], 1) #Append deterministic CON initial conditions conditional on parameter values to C path. 
            log_lik = neg_log_lik_scon(C_vector, T_span_tensor.to(device), dt, I_S_tensor.to(device), I_D_tensor.to(device), drift_diffusion_scon, scon_params_dict, temp_ref)
            ELBO = log_prob.mean() + log_lik.mean() - obs_model(C_vector)
            best_loss = ELBO if ELBO < best_loss else best_loss
            ELBO.backward()
            losses.append(ELBO.item())
            if len(losses) > 10:
                losses.pop(0)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
            optimizer.step()
            if iter % 20 == 0:
                print(f"Moving average loss at {iter} iterations is: {sum(losses) / len(losses)}. Best loss values is: {best_loss}.")
                print(C_vector.mean(-2))
                print('\n C_vector =', C_vector)
            if iter % 100000 == 0 and iter > 0:
                optimizer.param_groups[0]['lr'] *= 0.1
            t.update()

train_scon(100000, 1, T_span_tensor, I_S_tensor, I_D_tensor, scon_params_dict)
