import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

#Torch-related imports
import torch
import torch.distributions as D
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from TruncatedNormal import *

torch.manual_seed(0)
np.random.seed(0)

print('cuda device available?: ', torch.cuda.is_available())
active_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

torch.set_printoptions(precision = 6)

error_scale = 0.1

#SCON theta truncated normal distribution parameters in order of mean, sdev, lower, and upper.
u_M_details = torch.Tensor([0.001, 0.001 * error_scale, 0, 0.01])
a_SD_details = torch.Tensor([0.5, 0.5 * error_scale, 0, 1])
a_DS_details = torch.Tensor([0.5, 0.5 * error_scale, 0, 1])
a_M_details = torch.Tensor([0.5, 0.5 * error_scale, 0, 1])
a_MSC_details = torch.Tensor([0.5, 0.5 * error_scale, 0, 1])
k_S_ref_details = torch.Tensor([0.0002, 0.0002 * error_scale, 0, 0.001])
k_D_ref_details = torch.Tensor([0.0008, 0.0008 * error_scale, 0, 0.001])
k_M_ref_details = torch.Tensor([0.0003, 0.0003 * error_scale, 0, 0.001])
Ea_S_details = torch.Tensor([55, 55 * error_scale, 20, 120])
Ea_D_details = torch.Tensor([48, 48 * error_scale, 20, 120])
Ea_M_details = torch.Tensor([48, 48 * error_scale, 20, 120])

#SCON-C diffusion matrix parameter distribution s
c_SOC_details = torch.Tensor([0.05, 0.05 * error_scale, 0, 0.1])
c_DOC_details = torch.Tensor([0.001, 0.001 * error_scale, 0, 0.01])
c_MBC_details = torch.Tensor([0.001, 0.001 * error_scale, 0, 0.01])

#SCON-C theta distribution objects
u_M_dist = TruncatedNormal(loc = u_M_details[0], scale = u_M_details[1], a = u_M_details[2], b = u_M_details[3])
a_SD_dist = TruncatedNormal(loc = a_SD_details[0], scale = a_SD_details[1], a = a_SD_details[2], b = a_SD_details[3])
a_DS_dist = TruncatedNormal(loc = a_DS_details[0], scale = a_DS_details[1], a = a_DS_details[2], b = a_DS_details[3])
a_M_dist = TruncatedNormal(loc = a_M_details[0], scale = a_M_details[1], a = a_M_details[2], b = a_M_details[3])
a_MSC_dist = TruncatedNormal(loc = a_MSC_details[0], scale = a_MSC_details[1], a = a_MSC_details[2], b = a_MSC_details[3])
k_S_ref_dist = TruncatedNormal(loc = k_S_ref_details[0], scale = k_S_ref_details[1], a = k_S_ref_details[2], b = k_S_ref_details[3])
k_D_ref_dist = TruncatedNormal(loc = k_D_ref_details[0], scale = k_D_ref_details[1], a = k_D_ref_details[2], b = k_D_ref_details[3])
k_M_ref_dist = TruncatedNormal(loc = k_M_ref_details[0], scale = k_M_ref_details[1], a = k_M_ref_details[2], b = k_M_ref_details[3])
Ea_S_dist = TruncatedNormal(loc = Ea_S_details[0], scale = Ea_S_details[1], a = Ea_S_details[2], b = Ea_S_details[3])
Ea_D_dist = TruncatedNormal(loc = Ea_D_details[0], scale = Ea_D_details[1], a = Ea_D_details[2], b = Ea_D_details[3])
Ea_M_dist = TruncatedNormal(loc = Ea_M_details[0], scale = Ea_M_details[1], a = Ea_M_details[2], b = Ea_M_details[3])
c_SOC_dist = TruncatedNormal(loc = c_SOC_details[0], scale = c_SOC_details[1], a = c_SOC_details[2], b = c_SOC_details[3])
c_DOC_dist = TruncatedNormal(loc = c_DOC_details[0], scale = c_DOC_details[1], a = c_DOC_details[2], b = c_DOC_details[3])
c_MBC_dist = TruncatedNormal(loc = c_MBC_details[0], scale = c_MBC_details[1], a = c_MBC_details[2], b = c_MBC_details[3])

#SCON-C theta rsample draws
u_M = u_M_dist.rsample()
a_SD = a_SD_dist.rsample()
a_DS = a_DS_dist.rsample()
a_M = a_M_dist.rsample()
a_MSC = a_MSC_dist.rsample()
k_S_ref = k_S_ref_dist.rsample()
k_D_ref = k_D_ref_dist.rsample()
k_M_ref = k_M_ref_dist.rsample()
Ea_S = Ea_S_dist.rsample()
Ea_D = Ea_D_dist.rsample()
Ea_M = Ea_M_dist.rsample()
c_SOC = c_SOC_dist.rsample()
c_DOC = c_DOC_dist.rsample()
c_MBC = c_MBC_dist.rsample()

theta_dict = {'u_M': u_M, 'a_SD': a_SD, 'a_DS': a_DS, 'a_M': a_M, 'k_S_ref': k_S_ref, 'k_D_ref': k_D_ref, 'k_M_ref': k_M_ref, 'Ea_S': Ea_S, 'Ea_D': Ea_D, 'Ea_M': Ea_M, 'c_SOC': c_SOC, 'c_DOC': c_DOC, 'c_MBC': c_MBC}
print(theta_dict)


#SCON-C log prob evaluations
u_M_lp = u_M_dist.log_prob(u_M)
a_SD_lp = a_SD_dist.log_prob(a_SD)
a_DS_lp = a_DS_dist.log_prob(a_DS)
a_M_lp = a_M_dist.log_prob(a_M)
k_S_ref_lp = k_S_ref_dist.log_prob(k_S_ref)
k_D_ref_lp = k_D_ref_dist.log_prob(k_D_ref)
k_M_ref_lp = k_M_ref_dist.log_prob(k_M_ref)
Ea_S_lp = Ea_S_dist.log_prob(Ea_S)
Ea_D_lp = Ea_D_dist.log_prob(Ea_D)
Ea_M_lp = Ea_M_dist.log_prob(Ea_M)
c_SOC_lp = c_SOC_dist.log_prob(c_SOC)
c_DOC_lp = c_DOC_dist.log_prob(c_DOC)
c_MBC_lp = c_MBC_dist.log_prob(c_MBC)

lp_dict = {'u_M_lp': u_M_lp, 'a_SD_lp': a_SD_lp, 'a_DS_lp': a_DS_lp, 'a_M_lp': a_M_lp, 'k_S_ref_lp': k_S_ref_lp, 'k_D_ref_lp': k_D_ref_lp, 'k_M_ref_lp': k_M_ref, 'Ea_S_lp': Ea_S_lp, 'Ea_D_lp': Ea_D_lp, 'Ea_M_lp': Ea_M_lp, 'c_SOC_lp': c_SOC_lp, 'c_DOC_lp': c_DOC_lp, 'c_MBC_lp': c_MBC_lp}

print(lp_dict)
