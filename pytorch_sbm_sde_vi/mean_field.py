#Torch-related imports
import torch
from torch import nn
import torch.distributions as D
from torch.autograd import Function

#Module imports
from obs_and_flow import LowerBound
from TruncatedNormal import *
from LogitNormal import *

'''
This module defines the MeanField class for mean field VI inference of the soil biogeochemical model SDE system parameters.
It accepts priors stored in a dictionary with parameter value names as the key and statistical distribution parameter values.
Posterior samples and log_q_theta are output.
'''

class MeanField(nn.Module):

    '''
    Class for mean-field variational inference of SBM SDE parameters.
    Takes dictionary of parameter distribution information in order of mean, sdev, upper bound, and lower bound.

    The forward method returns dict_out, samples, log_q_theta, dict_parent_loc_scale.
    dict_out: dictionary of theta samples corresponding to their parameter keys for debugging and printing.
    samples: tensor of the theta values themselves for inference use.
    log_q_theta: log probability of the theta values corresponding to the variational distribution.
    dict_parent_loc_scale: dictionary of the values of the parent loc and scale distribution parameters.

    Formerly, the forward method returned dict_mean_sd, a dictionary of the values of the transformed distribution means and standard deviations, but calculation of the distribution mean and standard deviation was computationally wasteful to compute at every iteration and simpler for the TruncatedNormal distribution class than RescaledLogitNormal. The means and standard deviations can be computed from dict_parent_loc_scale.
    '''

    def __init__(self, DEVICE, PARAM_NAMES, INIT_DICT, DIST_CLASS, LEARN_COV=False):
        super().__init__()
        #Use param dict to intialise the means for the mean-field approximations.
        #init_params: name -> (parent mean, parent sd, true lower, true upper)
        means = []
        sds = []
        lower_bounds = []        
        upper_bounds = []
        for key in PARAM_NAMES:
            mean, sd, lower, upper = INIT_DICT[key]
            means.append(mean)
            sds.append(sd)
            upper_bounds.append(upper)
            lower_bounds.append(lower)

        self.dist = DIST_CLASS
        self.learn_cov = LEARN_COV
        self.means = nn.Parameter(torch.Tensor(means).to(DEVICE))
        if LEARN_COV:
            # Since we init post with indep priors, scale_tril = diag(stddev)
            scale_tril = torch.diag(torch.Tensor(sds)).to(DEVICE)
            # Map scale_tril to unconstrained space
            unconstrained_scale_tril = D.transform_to(self.dist.arg_constraints['scale_tril']).inv(scale_tril)
            self.sds = nn.Parameter(unconstrained_scale_tril) # (num_params, num_params)
        else:
            self.sds = nn.Parameter(torch.Tensor(sds).to(DEVICE)) # (num_params, )
        self.lowers = torch.Tensor(lower_bounds).to(DEVICE)
        self.uppers = torch.Tensor(upper_bounds).to(DEVICE)
        
        #Save keys for forward output.
        self.keys = PARAM_NAMES

    def forward(self, N = 10): # N should be assigned batch size in `train` function from training.py.
        #Update posterior.
        parent_loc = self.means
        if self.learn_cov:
            parent_scale_tril = D.transform_to(self.dist.arg_constraints['scale_tril'])(self.sds)
            parent_scale = torch.diag(parent_scale_tril) # this is incorrect unless indep, but not used in inference
            q_dist = self.dist(parent_loc, scale_tril=parent_scale_tril, a = self.lowers, b = self.uppers)
        else:
            if self.dist == TruncatedNormal:
                parent_loc = LowerBound.apply(self.means, self.lowers)
            parent_scale = LowerBound.apply(self.sds, 1e-8)
            q_dist = self.dist(parent_loc, parent_scale, a = self.lowers, b = self.uppers)

        # Sample theta ~ q(theta).
        samples = q_dist.rsample([N]) # (N, num_params)
        
        # Evaluate log prob of theta samples.
        if self.learn_cov:
            log_q_theta = q_dist.log_prob(samples) # (N, )
        else:
            log_q_theta = torch.sum(q_dist.log_prob(samples), -1) # (N, )
        
        # Return samples in same dictionary format.
        dict_out = {} #Define dictionary with n samples for each parameter.
        for key, sample in zip(self.keys, torch.split(samples, 1, -1),):
            dict_out[f'{key}'] = sample.squeeze(1) #Each sample is of shape [n].
        
        dict_parent_loc_scale = {} #Define dictionary to store parent parameter normal distribution means and standard deviations.
        #dict_mean_sd = {} #Define dictionary to store real parameter normal distribution means and standard deviations for TruncatedNormal distribution.
        #real_loc = q_dist.mean #q_dist._mean
        #real_scale = q_dist.stddev #torch.sqrt(q_dist._variance)
        # for key, parent_loc_scale, mean_sd in zip(self.keys, torch.split(torch.stack([parent_loc, parent_scale], 1), 1, 0), torch.split(torch.stack([real_loc, real_scale], 1), 1, 0)):
        #     dict_parent_loc_scale[f'{key}'] = parent_loc_scale
        #     dict_mean_sd[f'{key}'] = mean_sd
        for key, parent_loc_scale in zip(self.keys, torch.split(torch.stack([parent_loc, parent_scale], 1), 1, 0)):
            dict_parent_loc_scale[f'{key}'] = parent_loc_scale
        
        #Return samples in dictionary and tensor format.                                
        #return dict_out, samples, log_q_theta, dict_parent_loc_scale, dict_mean_sd
        return dict_out, samples, log_q_theta, dict_parent_loc_scale

        #if self.dist == TruncatedNormal:
        #    dict_mean_sd = {} #Define dictionary to store real parameter normal distribution means and standard deviations for TruncatedNormal distribution.
        #    real_loc = q_dist.mean #q_dist._mean
        #    real_scale = q_dist.stddev #torch.sqrt(q_dist._variance)
        #    for key, loc_scale, mean_sd in zip(self.keys, torch.split(torch.stack([parent_loc, parent_scale], 1), 1, 0), torch.split(torch.stack([real_loc, real_scale], 1), 1, 0)):
        #        dict_parent_loc_scale[f'{key}'] = loc_scale
        #        dict_mean_sd[f'{key}'] = mean_sd
        #    #Return samples in dictionary and tensor format.                                
        #    return dict_out, samples, log_q_theta, dict_parent_loc_scale, dict_mean_sd

        #elif self.dist == RescaledLogitNormal:
        #    for key, loc_scale in zip(self.keys, torch.split(torch.stack([parent_loc, parent_scale], 1), 1, 0)):
        #        dict_parent_loc_scale[f'{key}'] = loc_scale
        #    #Return samples in dictionary and tensor format.                
        #    return dict_out, samples, log_q_theta, dict_parent_loc_scale        
