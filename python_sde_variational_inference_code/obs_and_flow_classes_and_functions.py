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

def csv_to_obs_df_and_error(df_csv_string, STATE_DIM, error_scale, T):
    '''
    Takes CSV of labeled biogeochemical data observations and returns three items: 
    1) Numpy array of observation measurement times.
    2) Observations tensor including observations up to desired experiment hour threshold. 
    3) Observation error standard deviation at desired proportion of mean observation values. 
    '''
    obs_times = np.array(obs_df_con['hour'])    
    obs_df_full = pd.read_csv('df_csv_string', T)
    obs_df = obs_df_full[obs_df_full['hour'] <= T]
    obs_means = torch.Tensor(np.array(obs_df.drop(columns = 'hour')))    
    obs_df_T = obs_means.T
    obs_error_sd = torch.mean(obs_means_con_T).reshape([1, STATE_DIM])
    return obs_times, obs_df_T, obs_error_sd
