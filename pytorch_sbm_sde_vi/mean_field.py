import torch
from torch import nn
from obs_and_flow_classes_and_functions import LowerBound
import torch.distributions as D

'''
This module defines the MeanField class for mean field VI inference of the soil biogeochemical model SDE system parameters.
It accepts priors stored in a dictionary with parameter value names as the key and statistical distribution parameter values.
Posterior samples and log_q_theta are output.
'''

class MeanField(nn.Module):
    def __init__(self, init_params, sdev_scale_factor):
        super().__init__()

        #Use param dict to intialise the means for the mean-field approximations.
        means = []
        keys = []
        for key, value in init_params.items():
            keys += [key]
            means += [value]
        self.means = nn.Parameter(torch.Tensor(means))
        self.sds = nn.Parameter(self.means * sdev_scale_factor)
        #Save keys for forward output.
        self.keys = keys

    def forward(self, n = 10):
        #Update posterior.
        q_dist = D.normal.Normal(LowerBound.apply(self.means, 1e-6), LowerBound.apply(self.sds, 1e-8)) #Clamping distribution mean values a bit above 0. 
        #Sample theta ~ q(theta).
        samples = LowerBound.apply(q_dist.rsample([n]), 1e-6) #Clamping sample values a bit above 0.
        #Evaluate log prob of theta samples.
        log_q_theta = torch.sum(q_dist.log_prob(samples), -1) #Shape of n.
        #Return samples in same dictionary format.
        dict_out = {} #Define dictionary with n samples for each parameter.
        for key, sample in zip(self.keys, torch.split(samples, 1, -1),):
            dict_out[f"{key}"] = sample.squeeze(1) #Each sample is of shape [n].
        #Return samples in dictionary and tensor format.
        return dict_out, samples, log_q_theta
