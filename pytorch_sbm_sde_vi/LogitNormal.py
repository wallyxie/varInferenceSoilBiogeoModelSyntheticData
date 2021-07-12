import torch
from torch import nn
from obs_and_flow import LowerBound
import torch.distributions as D
from torch.distributions.utils import broadcast_all

def logit(x, lower=0, upper=1):
    return torch.logit((x - lower) / (upper - lower))

class RescaledLogitNormal:
    def __init__(self, loc=0, scale=1, a=0, b=1):
        # loc: mean of the normally distributed logit
        # scale: standard deviation of the normally distributed logit
        # a, b: lower and upper bounds, respectively
        loc, scale, a, b = broadcast_all(loc, scale, a, b)
        self.sigmoid = RescaledSigmoid(a, b)
        #print(self.logit(sigmoid_loc))
        self.base = D.normal.Normal(loc, scale)

    def logit(self, x):
        lower, upper = self.sigmoid.lower, self.sigmoid.upper
        return logit(x, lower, upper)

    def rsample(self, shape):
        logit_x = self.base.rsample(shape)
        x = self.sigmoid(logit_x)
        return x

    def log_prob(self, x):
        lower, upper = self.sigmoid.lower, self.sigmoid.upper
        logit_x = self.logit(x)
        jac = (upper - lower)/((x-lower)*(upper-x))
        return self.base.log_prob(logit_x) + torch.log(jac)

class RescaledSigmoid(nn.Module):
    def __init__(self, lower, upper):
        super().__init__()
        self.upper = upper
        self.lower = lower

    def forward(self, x, eps=1e-5):
        y = 1/(1+torch.exp(-x))
        y = (self.upper-self.lower)*y+self.lower
        y = LowerBound.apply(y, self.lower+eps)
        y = -LowerBound.apply(-y, -self.upper+eps)
        return y
