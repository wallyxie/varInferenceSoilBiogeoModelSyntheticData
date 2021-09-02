#PyData imports
import numpy as np

#Torch-related imports
import torch

#Module-specific imports
from obs_and_flow import LowerBound

'''
This script includes the linear and Arrhenius temperature dependence functions to induce temperature-based forcing in differential equation soil biogeochemical models (SBMs). It also includes the SBM SDE classes corresponding to the various parameterizations of the stochastic conventional (SCON), stochastic AWB (SAWB), and stochastic AWB-equilibrium chemistry approximation (SAWB-ECA) for incorporation with normalizing flow "neural stochastic differential equation" solvers. The following SBM SDE system parameterizations are contained in this script:
    1) SCON constant diffusion (SCON-c)
    2) SCON state scaling diffusion (SCON-ss)
    3) SAWB constant diffusion (SAWB-c)
    4) SAWB state scaling diffusion (SAWB-ss)
    5) SAWB-ECA constant diffusion (SAWB-ECA-c)
    6) SAWB-ECA state scaling diffusion (SAWB-ECA-ss)
The respective analytical steady state estimation functions derived from the deterministic ODE versions of the stochastic SBMs are no longer included in this script, as we are no longer initiating SBMs at steady state before starting simulations.
'''

############################################################
##SOIL BIOGEOCHEMICAL MODEL TEMPERATURE RESPONSE FUNCTIONS##
############################################################

def temp_gen(t: torch.Tensor, TEMP_REF: float, TEMP_RISE: float = 5) -> torch.Tensor:
    '''
    Temperature function to force soil biogeochemical models.
    Accepts input time(s) t in torch.Tensor type.
    This particular temperature function assumes soil temperatures will increase by TEMP_REF over the next 80 years.    
    Returns a tensor of one or more temperatures in K given t.
    '''
    temp = TEMP_REF + (TEMP_RISE * t) / (80 * 24 * 365) + 10 * torch.sin((2 * np.pi / 24) * t) + 10 * torch.sin((2 * np.pi / (24 * 365)) * t)
    return temp

def arrhenius_temp_dep(parameter, temp: float, Ea: torch.Tensor, TEMP_REF: float) -> torch.Tensor:
    '''
    Arrhenius temperature dependence function.
    Accepts input parameter as torch.Tensor or Python float type.
    Accepts Ea as torch.Tensor type only.
    0.008314 is the gas constant. Temperatures are in K.
    Returns a tensor of transformed parameter value(s).    
    '''
    decayed_parameter = parameter * torch.exp(-Ea / 0.008314 * (1 / temp - 1 / TEMP_REF))
    return decayed_parameter

def linear_temp_dep(parameter, temp, Q, TEMP_REF):
    '''
    For a parameter with linear temperature dependence, returns the transformed parameter value.
    Q is the slope of the temperature dependence and is a varying parameter.
    Temperatures are in K.
    '''
    modified_parameter = parameter - Q * (temp - TEMP_REF)
    return modified_parameter

##########################
##LITTER INPUT FUNCTIONS##
##########################

def i_s(t):
    '''
    This is the endogenous SOC litter input function.
    '''
    return 0.001 + 0.0005 * torch.sin((2 * np.pi / (24 * 365)) * t)

def i_d(t):
    '''
    This is the endogenous DOC litter input function.
    '''
    return 0.0001 + 0.00005 * torch.sin((2 * np.pi / (24 * 365)) * t)

##################################################
##STOCHASTIC DIFFERENTIAL EQUATION MODEL CLASSES##
##################################################

class SBM_SDE:
    '''
    This is the 
    '''

    def __init__(
            self,
            C_PATH, T_SPAN_TENSOR, I_S_TENSOR, I_D_TENSOR, TEMP_TENSOR, TEMP_REF):
        pass

class SCON(SBM_SDE):

    @staticmethod
    def drift_diffusion(C_PATH, T_SPAN_TENSOR, I_S_TENSOR, I_D_TENSOR, TEMP_TENSOR, TEMP_REF, SCONR_C_fix_u_M_a_Ea_c_params_dict, diffusion_type):

    @staticmethod
    def get_CO2(...):

class SAWB(SBM_SDE):

    @staticmethod
    def drift_diffusion(...):

    @staticmethod
    def get_CO2(...):

class SAWB_ECA(SBM_SDE):
    
    @staticmethod
    def drift_diffusion(...):

    @staticmethod
    def get_CO2(...):
