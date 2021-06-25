import torch
from torch import nn
from obs_and_flow import LowerBound, SoftplusLayer
import torch.distributions as D

'''
This module defines the MeanField class for mean field VI inference of the soil biogeochemical model SDE system parameters.
It accepts priors stored in a dictionary with parameter value names as the key and statistical distribution parameter values.
Posterior samples and log_q_theta are output.
'''

class BoundedSigmoid(nn.Module):

    def __init__(self, upper, lower):
        super().__init__()
        self.upper = upper
        self.lower = lower

    def forward(self, x):
        # in.shape == out.shape == (batch_size, 1, n * state_dim)
        y = self._fwd(x)
        ildj = self._ildj(y)
        return y, ildj

    def _fwd(self, x, eps=1e-5):
        y = 1/(1+torch.exp(-x))
        y = (self.upper-self.lower)*y+self.lower
        y = LowerBound.apply(y, self.lower+eps)
        y = -LowerBound.apply(-y, -self.upper+eps)
        return y

    def _ildj(self, y):
        ildj = (self.upper - self.lower)/((y-self.lower)*(self.upper-y))
        return torch.log(ildj)

class MeanField(nn.Module):
    def __init__(self, init_params, sdev_scale_factor):
        super().__init__()

        #Use param dict to intialise the means for the mean-field approximations.
        # init_params: name -> (init_value, lower, upper)
        keys = []
        means = []
        upper_bounds = []
        lower_bounds = []
        for key, (value, lower, upper) in init_params.items():
            keys.append(key)
            means.append(value)
            upper_bounds.append(upper)
            lower_bounds.append(lower)

        self.means = nn.Parameter(torch.Tensor(means))
        self.sds = nn.Parameter(self.means * sdev_scale_factor)
        self.sigmoid = BoundedSigmoid(upper_bounds, lower_bounds)
        
        #Save keys for forward output.
        self.keys = keys

    def forward(self, n = 10):
        #Update posterior.
        q_dist = D.normal.Normal(self.means, LowerBound.apply(self.sds, 1e-8)) #Clamping distribution mean values a bit above 0. 
        #Sample theta ~ q(theta).
        eps = q_dist.rsample([n]) # (num_samples, num_params)
        #Evaluate log prob of theta samples.
        log_q_theta = torch.sum(q_dist.log_prob(eps), -1) #Shape of n.

        # Apply sigmoid transformation
        samples, ildj = self.sigmoid(eps)
        log_q_theta += ildj.sum(-1)
        
        #Return samples in same dictionary format.
        dict_out = {} #Define dictionary with n samples for each parameter.
        for key, sample in zip(self.keys, torch.split(samples, 1, -1),):
            dict_out[f"{key}"] = sample.squeeze(1) #Each sample is of shape [n].
        #Return samples in dictionary and tensor format.
        return dict_out, samples, log_q_theta
