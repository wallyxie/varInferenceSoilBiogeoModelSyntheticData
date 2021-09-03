from typing import Dict, Tuple, Union

#PyData imports
import numpy as np

#Torch-related imports
import torch

#Module-specific imports
from obs_and_flow import LowerBound

'''
This script includes the linear and Arrhenius temperature dependence functions to induce temperature-based forcing in differential equation soil biogeochemical models (SBMs). It also includes the SBM SDE classes corresponding to the various parameterizations of the stochastic conventional (SCON), stochastic AWB (SAWB), and stochastic AWB-equilibrium chemistry approximation (SAWB-ECA) for incorporation with normalizing flow "neural stochastic differential equation" solvers. The following SBM SDE system parameterizations are contained in this script:
    1) SCON constant diffusion (SCON-C)
    2) SCON state scaling diffusion (SCON-SS)
    3) SAWB constant diffusion (SAWB-C)
    4) SAWB state scaling diffusion (SAWB-SS)
    5) SAWB-ECA constant diffusion (SAWB-ECA-C)
    6) SAWB-ECA state scaling diffusion (SAWB-ECA-SS)
The respective analytical steady state estimation functions derived from the deterministic ODE versions of the stochastic SBMs are no longer included in this script, as we are no longer initiating SBMs at steady state before starting simulations.
'''

DictOfTensors = Dict[str, torch.Tensor]
Number = Union[int, float]
TupleOfTensors = Tuple[torch.Tensor, torch.Tensor]

############################################################
##SOIL BIOGEOCHEMICAL MODEL TEMPERATURE RESPONSE FUNCTIONS##
############################################################

def temp_gen(t: torch.Tensor, TEMP_REF: Number, TEMP_RISE: Number = 5) -> torch.Tensor:
    '''
    Temperature function to force soil biogeochemical models.
    Accepts input time(s) t in torch.Tensor type.
    This particular temperature function assumes soil temperatures will increase by TEMP_REF over the next 80 years.    
    Returns a tensor of one or more temperatures in K given t.
    '''
    temp = TEMP_REF + (TEMP_RISE * t) / (80 * 24 * 365) + 10 * torch.sin((2 * np.pi / 24) * t) + 10 * torch.sin((2 * np.pi / (24 * 365)) * t)
    return temp

def arrhenius_temp_dep(parameter, temp: Number, Ea: torch.Tensor, TEMP_REF: Number) -> torch.Tensor:
    '''
    Arrhenius temperature dependence function.
    Accepts input parameter as torch.Tensor or Python Number type.
    Accepts Ea as torch.Tensor type only.
    0.008314 is the gas constant. Temperatures are in K.
    Returns a tensor of transformed parameter value(s).    
    '''
    decayed_parameter = parameter * torch.exp(-Ea / 0.008314 * (1 / temp - 1 / TEMP_REF))
    return decayed_parameter

def linear_temp_dep(parameter, temp: Number, Q: torch.Tensor, TEMP_REF: Number) -> torch.Tensor:
    '''
    For a parameter with linear temperature dependence, returns the transformed parameter value.
    Accepts input parameter as torch.Tensor or Python Number type.    
    Q is the slope of the temperature dependence and is a varying parameter.
    Temperatures are in K.
    '''
    modified_parameter = parameter - Q * (temp - TEMP_REF)
    return modified_parameter

##########################
##LITTER INPUT FUNCTIONS##
##########################

def i_s(t: torch.Tensor) -> torch.Tensor:
    '''
    This is the endogenous SOC litter input function.
    '''
    return 0.001 + 0.0005 * torch.sin((2 * np.pi / (24 * 365)) * t)

def i_d(t: torch.Tensor) -> torch.Tensor:
    '''
    This is the endogenous DOC litter input function.
    '''
    return 0.0001 + 0.00005 * torch.sin((2 * np.pi / (24 * 365)) * t)

##################################################
##STOCHASTIC DIFFERENTIAL EQUATION MODEL CLASSES##
##################################################

class SBM_SDE:
    '''
    This is the base class for evaluating the SBM SDE SSMs. Acronyms, acronyms, acronyms!
    '''

    def __init__(
            self,
            T_SPAN_TENSOR: torch.Tensor, 
            I_S_TENSOR: torch.Tensor, 
            I_D_TENSOR: torch.Tensor, 
            TEMP_TENSOR: torch.Tensor, 
            TEMP_REF: Number
            ):

        self.times = T_SPAN_TENSOR
        self.i_S = I_S_TENSOR
        self.i_D = I_D_TENSOR
        self.temps = TEMP_TENSOR
        self.temp_ref = TEMP_REF

class SCON(SBM_SDE):
    '''
    Class contains SCON SDE drift (alpha) and diffusion (beta) equations.
    Constant (C) and state-scaling (SS) diffusion paramterizations are included. diffusion_type must thereby be specified as 'C' or 'SS'. 
    Other diffusion parameterizations are not included.
    '''
    def __init__(
            self,
            T_SPAN_TENSOR: torch.Tensor, 
            I_S_TENSOR: torch.Tensor, 
            I_D_TENSOR: torch.Tensor, 
            TEMP_TENSOR: torch.Tensor, 
            TEMP_REF: Number,
            diffusion_type: str
            ):
        super().__init__(T_SPAN_TENSOR, I_S_TENSOR, I_D_TENSOR, TEMP_TENSOR, TEMP_REF)

        if diffusion_type not in {'C', 'SS'}:
            raise NotImplementedError('Other diffusion parameterizations aside from constant (c) or state-scaling (ss) have not been implemented.')

        self.diffusion_type = diffusion_type
        self.state_dim = 3

    def drift_diffusion(
            self,
            C_PATH: torch.Tensor, 
            SCON_params_dict: DictOfTensors, 
            ) -> TupleOfTensors:
        '''
        Returns SCON drift and diffusion tensors 
        Expected SCON_params_dict = {'u_M': u_M, 'a_SD': a_SD, 'a_DS': a_DS, 'a_M': a_M, 'a_MSC': a_MSC, 'k_S_ref': k_S_ref, 'k_D_ref': k_D_ref, 'k_M_ref': k_M_ref, 'Ea_S': Ea_S, 'Ea_D': Ea_D, 'Ea_M': Ea_M, '[cs]_SOC': [cs]_SOC, '[cs]_DOC': [cs]_DOC, '[cs]_MBC': [cs]_MBC}
        '''
        #Partition SOC, DOC, MBC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor. 
        SOC, DOC, MBC =  torch.chunk(C_PATH, self.state_dim, -1)
        #Initiate tensor with same dims as C_PATH to assign drift.
        drift = torch.empty_like(C_PATH, device = C_PATH.device)
        #Decay parameters are forced by temperature changes.
        k_S = arrhenius_temp_dep(SCON_params_dict['k_S_ref'], self.temps, SCON_params_dict['Ea_S'], self.temp_ref) #Apply vectorized temperature-dependent transformation to k_S_ref.
        k_S = k_S.permute(2, 1, 0) #Get k_S into appropriate dimensions. 
        k_D = arrhenius_temp_dep(SCON_params_dict['k_D_ref'], self.temps, SCON_params_dict['Ea_D'], self.temp_ref) #Apply vectorized temperature-dependent transformation to k_D_ref.
        k_D = k_D.permute(2, 1, 0) #Get k_D into appropriate dimensions.
        k_M = arrhenius_temp_dep(SCON_params_dict['k_M_ref'], self.temps, SCON_params_dict['Ea_M'], self.temp_ref) #Apply vectorized temperature-dependent transformation to k_M_ref.
        k_M = k_M.permute(2, 1, 0) #Get k_M into appropriate dimensions.
        #Repeat and permute parameter values to match dimension sizes
        SCON_params_dict_rep = dict((k, v.repeat(1, self.times.size(1), 1).permute(2, 1, 0)) for k, v in SCON_params_dict.items())    
        #Drift is calculated.
        drift_SOC = self.i_S + SCON_params_dict_rep['a_DS'] * k_D * DOC + SCON_params_dict_rep['a_M'] * SCON_params_dict_rep['a_MSC'] * k_M * MBC - k_S * SOC
        drift_DOC = self.i_D + SCON_params_dict_rep['a_SD'] * k_S * SOC + SCON_params_dict_rep['a_M'] * (1 - SCON_params_dict_rep['a_MSC']) * k_M * MBC - (SCON_params_dict_rep['u_M'] + k_D) * DOC
        drift_MBC = SCON_params_dict_rep['u_M'] * DOC - k_M * MBC
        #Diffusion matrix is computed based on diffusion type.
        diffusion_sqrt = torch.zeros([drift.size(0), drift.size(1), self.state_dim, self.state_dim], device = drift.device) #Create tensor to assign diffusion matrix elements.        
        if self.diffusion_type == 'C':
            diffusion_sqrt_single = torch.diag_embed(torch.sqrt(torch.stack([LowerBound.apply(SCON_params_dict['c_SOC'], 1e-8), LowerBound.apply(SCON_params_dict['c_DOC'], 1e-8), LowerBound.apply(SCON_params_dict['c_MBC'], 1e-8)], 1)))
            diffusion_sqrt = diffusion_sqrt_single.unsqueeze(1).expand(-1, self.times.size(1), -1, -1) #Expand diffusion matrices across all paths and across discretized time steps.            
        elif self.diffusion_type == 'SS':
            diffusion_sqrt = torch.zeros([drift.size(0), drift.size(1), self.state_dim, self.state_dim], device = drift.device) #Create tensor to assign diffusion matrix elements.            
            diffusion_sqrt[:, :, 0 : 1, 0] = torch.sqrt(LowerBound.apply(SOC * SCON_params_dict_rep['s_SOC'], 1e-8)) #SOC diffusion standard deviation
            diffusion_sqrt[:, :, 1 : 2, 1] = torch.sqrt(LowerBound.apply(DOC * SCON_params_dict_rep['s_DOC'], 1e-8)) #DOC diffusion standard deviation
            diffusion_sqrt[:, :, 2 : 3, 2] = torch.sqrt(LowerBound.apply(MBC * SCON_params_dict_rep['s_MBC'], 1e-8)) #MBC diffusion standard deviation
        
        return drift, diffusion_sqrt

    def add_CO2():

class SAWB(SBM_SDE):

    @staticmethod
    def drift_diffusion(...):

    @staticmethod
    def add_CO2(...):

class SAWB_ECA(SBM_SDE):
    
    @staticmethod
    def drift_diffusion(...):

    @staticmethod
    def add_CO2(...):
