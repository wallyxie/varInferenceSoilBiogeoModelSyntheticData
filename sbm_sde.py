import torch as tc
import numpy as np

form sbm_temp_functions import *

####################################################
##STOCHASTIC DIFFERENTIAL EQUATION MODEL FUNCTIONS##
####################################################

def litter_scon(t):
    '''
    Returns SCON system exogenous litter input vector. Time t is on hourly scale.
    '''
    I_S = 0.001 + 0.0005 * np.sin((2 * pi / (24 * 365)) * t) #Exogenous SOC input function
    I_D =  0.0001 + 0.00005 * sin((2 * pi / (24 * 365)) * t) #Exogenous DOC input function
    litter_vector = tc.reshape(tc.FloatTensor([I_S, I_D, 0]), [3, 1])
    return litter_vector

def alpha_scon(c_vector, params_dict):
    '''
    Returns SCON system drift vector for approximate p(x).
    c_vector[0] is expected to be SOC, c_vector[1] is expected to be DOC, and c_vector[2] is expected to be MBC.
    '''
    c_vector = SOC, DOC, MBC
    SOC_drift = a_DS * k_D * DOC + a_M * a_MSC * k_M * MBC - k_S * SOC
    DOC_drift = a_SD * k_S * SOC + a_M * (1 - a_MSC) * k_M * MBC - (u_M + k_D) * DOC 
    MBC_drift = u_M * DOC - k_M * MBC
    drift_vector = tc.cat(
    return drift_vector
    #IN PROGRESS

def beta_scon():
    '''
    Returns SCON system diffusion matrix.
    '''
    #IN PROGRESS
    return
