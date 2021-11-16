import torch
from torch import nn
from obs_and_flow import LowerBound
import torch.distributions as D
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all, lazy_property
import numpy as np

def logit(x, lower=0, upper=1):
    rescaled_x = 1.0 * (x - lower) / (upper - lower)
    return torch.log(rescaled_x) - torch.log1p(-rescaled_x)

class RescaledLogitNormal(Distribution):
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive,
                       'a': constraints.real, 'b': constraints.real}
    has_rsample = True

    def __init__(self, loc=0, scale=1, a=0, b=1, validate_args=None):
        # loc: mean of the normally distributed logit
        # scale: standard deviation of the normally distributed logit
        # a, b: lower and upper bounds, respectively
        loc, scale, a, b = broadcast_all(loc, scale, a, b)
        self.sigmoid = RescaledSigmoid(a, b)
        self.base = D.normal.Normal(loc, scale)
        self.loc = self.base.loc
        self.scale = self.base.scale
        self.a = self.sigmoid.lower
        self.b = self.sigmoid.upper
        super().__init__(self.base.batch_shape, validate_args=validate_args)

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @lazy_property
    def mean(self):
        return self.approx_moment(1)

    @lazy_property
    def variance(self):
        return self.approx_moment(2) - self.mean**2

    def approx_moment(self, d=1, num_partitions=100000, eps=1e-8):
        lower, upper = self.sigmoid.lower + eps, self.sigmoid.upper - eps
        x = torch.from_numpy(np.linspace(lower, upper, num_partitions)) # (num_partitions, batch_shape)
        y = x**d * torch.exp(self.log_prob(x))
        y[torch.isnan(y)] = 0.0
        return torch.trapz(y, x, dim=0)

    def logit(self, x):
        lower, upper = self.sigmoid.lower, self.sigmoid.upper
        return logit(x, lower, upper)

    def rsample(self, sample_shape=torch.Size()):
        logit_x = self.base.rsample(sample_shape)
        x = self.sigmoid(logit_x)
        return x

    def log_prob(self, x):
        lower, upper = self.sigmoid.lower, self.sigmoid.upper
        logit_x = self.logit(x)
        jac = (upper - lower)/((x-lower)*(upper-x))
        return self.base.log_prob(logit_x) + torch.log(jac)

class MultivariateLogitNormal(Distribution):
    arg_constraints = {'loc': constraints.real_vector,
                       'covariance_matrix': constraints.positive_definite,
                       'precision_matrix': constraints.positive_definite,
                       'scale_tril': constraints.lower_cholesky,
                       'a': constraints.real_vector, 'b': constraints.real_vector}
    has_rsample = True

    def __init__(self, loc, covariance_matrix=None, precision_matrix=None, scale_tril=None,
                 a=0, b=1, validate_args=None):
        # loc: mean of the normally distributed logit
        # covariance_matrix, precision matrix, scale_tril: 
        #   of the normally distributed logit, exactly one must be provided
        # a, b: lower and upper bounds, respectively
        loc, a, b = broadcast_all(loc, a, b)
        self.sigmoid = RescaledSigmoid(a, b)
        self.base = D.multivariate_normal.MultivariateNormal(loc, 
            covariance_matrix, precision_matrix, scale_tril)
        
        self.loc = self.base.loc
        self.a = self.sigmoid.lower
        self.b = self.sigmoid.upper
        if scale_tril is not None:
            self.scale_tril = self.base.scale_tril
        elif covariance_matrix is not None:
            self.covariance_matrix = self.base.covariance_matrix
        else:
            self.precision_matrix = self.base.precision_matrix
        
        #print(self.base.batch_shape, self.base.event_shape, validate_args)
        super().__init__(self.base.batch_shape, self.base.event_shape,
                         validate_args=validate_args)

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @lazy_property
    def scale_tril(self): # of parent multivariate normal
        return self.base.scale_tril

    @lazy_property
    def covariance_matrix(self): # of parent multivariate normal
        return self.base.covariance_matrix

    @lazy_property
    def precision_matrix(self): # of parent multivariate normal
        return self.base.precision_matrix

    @lazy_property
    def mean(self):
        raise NotImplementedError
        #return self.approx_moment(1)

    @lazy_property
    def variance(self):
        raise NotImplementedError
        #return self.approx_moment(2) - self.mean**2

    def approx_moment(self, d=1, num_partitions=10000, eps=1e-8):
        raise NotImplementedError
        #lower, upper = self.sigmoid.lower + eps, self.sigmoid.upper - eps
        #x = torch.from_numpy(np.linspace(lower, upper, num_partitions)) # (num_partitions, batch_shape)
        #y = x**d * torch.exp(self.log_prob(x))
        #y[torch.isnan(y)] = 0.0
        #return torch.trapz(y, x, dim=0)

    def logit(self, x):
        lower, upper = self.sigmoid.lower, self.sigmoid.upper
        return logit(x, lower, upper)

    def rsample(self, sample_shape=torch.Size()):
        logit_x = self.base.rsample(sample_shape)
        x = self.sigmoid(logit_x)
        return x

    def log_prob(self, x):
        lower, upper = self.sigmoid.lower, self.sigmoid.upper
        logit_x = self.logit(x)
        log_jac = torch.log((upper - lower)/((x-lower)*(upper-x))).sum(-1) # sum across event_dim
        return self.base.log_prob(logit_x) + log_jac

class RescaledSigmoid(nn.Module):
    def __init__(self, lower, upper):
        super().__init__()
        self.upper = upper
        self.lower = lower

    def forward(self, x, eps=1e-8):
        y = 1/(1+torch.exp(-x))
        y = (self.upper-self.lower)*y+self.lower
        y = LowerBound.apply(y, self.lower+eps)
        y = -LowerBound.apply(-y, -self.upper+eps)
        return y
