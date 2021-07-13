#Torch-related imports
import torch
from torch import nn
import torch.distributions as D
from TruncatedNormal import *
from LogitNormal import *
from torch.autograd import Function

#Model-specific imports
from obs_and_flow import LowerBound

'''
This module defines the MeanField class for mean field VI inference of the soil biogeochemical model SDE system parameters.
It accepts priors stored in a dictionary with parameter value names as the key and statistical distribution parameter values.
Posterior samples and log_q_theta are output.
'''

class MeanField(nn.Module):

    '''
    Class for mean-field variational inference of SBM SDE parameters.
    Takes dictionary of parameter distribution information in order of
    mean, sdev, upper bound, and lower bound.
    '''

    def __init__(self, DEVICE, PARAM_NAMES, PRIOR_DIST_DETAILS_DICT, DIST_CLASS):
        super().__init__()
        #Use param dict to intialise the means for the mean-field approximations.
        #init_params: name -> (parent mean, parent sd, true lower, true upper)
        means = []
        sds = []
        lower_bounds = []        
        upper_bounds = []
        for key in PARAM_NAMES:
            mean, sd, lower, upper = PRIOR_DIST_DETAILS_DICT[key]
            means.append(mean)
            sds.append(sd)
            upper_bounds.append(upper)
            lower_bounds.append(lower)          

        self.dist = DIST_CLASS
        self.means = nn.Parameter(torch.Tensor(means).to(DEVICE))
        self.sds = nn.Parameter(torch.Tensor(sds).to(DEVICE))
        self.lowers = torch.Tensor(lower_bounds).to(DEVICE)
        self.uppers = torch.Tensor(upper_bounds).to(DEVICE)
        
        #Save keys for forward output.
        self.keys = PARAM_NAMES

    def forward(self, N = 10): # N should be assigned batch size in `train` function from training.py.
        #Update posterior.
        parent_loc = self.means
        parent_scale = LowerBound.apply(self.sds, 1e-6)
        q_dist = self.dist(parent_loc, parent_scale, a = self.lowers, b = self.uppers)

        # Sample theta ~ q(theta).
        samples = q_dist.rsample([N])
        
        # Evaluate log prob of theta samples.
        log_q_theta = torch.sum(q_dist.log_prob(samples), -1)
        
        # Return samples in same dictionary format.
        dict_out = {} #Define dictionary with n samples for each parameter.
        for key, sample in zip(self.keys, torch.split(samples, 1, -1),):
            dict_out[f'{key}'] = sample.squeeze(1) #Each sample is of shape [n].
        
        dict_parent_loc_scale = {} #Define dictionary to store parent parameter normal distribution means and standard deviations.

        if self.dist == TruncatedNormal:
            dict_real_loc_scale = {} #Define dictionary to store real parameter normal distribution means and standard deviations for TruncatedNormal distribution.
            real_loc = q_dist._mean
            real_scale = torch.sqrt(q_dist._variance)
            for key, loc_scale, real_loc_scale in zip(self.keys, torch.split(torch.stack([parent_loc, parent_scale], 1), 1, 0), torch.split(torch.stack([real_loc, real_scale], 1), 1, 0)):
                dict_parent_loc_scale[f'{key}'] = loc_scale
                dict_real_loc_scale[f'{key}'] = real_loc_scale
            #Return samples in dictionary and tensor format.                                
            return dict_out, samples, log_q_theta, dict_parent_loc_scale, dict_real_loc_scale

        elif self.dist == RescaledLogitNormal:
            for key, loc_scale in zip(self.keys, torch.split(torch.stack([parent_loc, parent_scale], 1), 1, 0)):
                dict_parent_loc_scale[f'{key}'] = loc_scale
            #Return samples in dictionary and tensor format.                
            return dict_out, samples, log_q_theta, dict_parent_loc_scale        
