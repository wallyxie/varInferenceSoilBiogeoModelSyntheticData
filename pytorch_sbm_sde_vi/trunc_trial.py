import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

#Torch-related imports
import torch
import torch.detailsributions as D
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
from TruncatedNormal import *

torch.manual_seed(0)
np.random.seed(0)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

torch.set_printoptions(precision = 8)

error_scale = 0.25

#SCON theta truncated normal detailsribution parameters in order of mean, sdev, lower, and upper.
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


