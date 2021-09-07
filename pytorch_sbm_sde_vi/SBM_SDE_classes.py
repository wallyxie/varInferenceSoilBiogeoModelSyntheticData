#Python-related imports
from typing import Dict, Tuple, Union

#PyData imports
import numpy as np

#Torch-related imports
import torch

#Module imports
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
    This is the base class for evaluating the SBM SDE SSMs.
    '''

    def __init__(self):
        pass

class SCON(SBM_SDE):
    '''
    Class contains SCON SDE drift (alpha) and diffusion (beta) equations.
    Constant (C) and state-scaling (SS) diffusion paramterizations are included. DIFFUSION_TYPE must thereby be specified as 'C' or 'SS'. 
    Other diffusion parameterizations are not included.
    '''
    def __init__(
            self,
            DIFFUSION_TYPE: str
            ):

        if DIFFUSION_TYPE not in {'C', 'SS'}:
            raise NotImplementedError('Other diffusion parameterizations aside from constant (c) or state-scaling (ss) have not been implemented.')

        self.DIFFUSION_TYPE = DIFFUSION_TYPE
        self.state_dim = 3

    def drift_diffusion(
        self,
        C_PATH: torch.Tensor, 
        SCON_params_dict: DictOfTensors,
        T_SPAN_TENSOR: torch.Tensor, 
        I_S_TENSOR: torch.Tensor, 
        I_D_TENSOR: torch.Tensor, 
        TEMP_TENSOR: torch.Tensor, 
        TEMP_REF: Number,
        ) -> TupleOfTensors:
        '''
        Accepts states x and dictionary of parameter samples.
        Returns SCON drift and diffusion tensors corresponding to state values and parameter samples.  
        Expected SCON_params_dict = {'u_M': u_M, 'a_SD': a_SD, 'a_DS': a_DS, 'a_M': a_M, 'a_MSC': a_MSC, 'k_S_ref': k_S_ref, 'k_D_ref': k_D_ref, 'k_M_ref': k_M_ref, 'Ea_S': Ea_S, 'Ea_D': Ea_D, 'Ea_M': Ea_M, '[cs]_SOC': [cs]_SOC, '[cs]_DOC': [cs]_DOC, '[cs]_MBC': [cs]_MBC}
        '''
        #Partition SOC, DOC, MBC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor. 
        SOC, DOC, MBC =  torch.chunk(C_PATH, self.state_dim, -1)
        #Repeat and permute parameter values to match dimension sizes.
        SCON_params_dict_rep = dict((k, v.repeat(1, T_SPAN_TENSOR.size(1), 1).permute(2, 1, 0)) for k, v in SCON_params_dict.items())
        #Initiate tensor with same dims as C_PATH to assign drift.
        drift = torch.empty_like(C_PATH, device = C_PATH.device)
        #Decay parameters are forced by temperature changes.
        k_S = arrhenius_temp_dep(SCON_params_dict_rep['k_S_ref'], TEMP_TENSOR, SCON_params_dict_rep['Ea_S'], TEMP_REF) #Apply vectorized temperature-dependent transformation to k_S_ref.
        k_D = arrhenius_temp_dep(SCON_params_dict_rep['k_D_ref'], TEMP_TENSOR, SCON_params_dict_rep['Ea_D'], TEMP_REF) #Apply vectorized temperature-dependent transformation to k_D_ref.
        k_M = arrhenius_temp_dep(SCON_params_dict_rep['k_M_ref'], TEMP_TENSOR, SCON_params_dict_rep['Ea_M'], TEMP_REF) #Apply vectorized temperature-dependent transformation to k_M_ref.
        #Drift is calculated.
        drift_SOC = I_S_TENSOR + SCON_params_dict_rep['a_DS'] * k_D * DOC + SCON_params_dict_rep['a_M'] * SCON_params_dict_rep['a_MSC'] * k_M * MBC - k_S * SOC
        drift_DOC = I_D_TENSOR + SCON_params_dict_rep['a_SD'] * k_S * SOC + SCON_params_dict_rep['a_M'] * (1 - SCON_params_dict_rep['a_MSC']) * k_M * MBC - (SCON_params_dict_rep['u_M'] + k_D) * DOC
        drift_MBC = SCON_params_dict_rep['u_M'] * DOC - k_M * MBC
        #Diffusion matrix is computed based on diffusion type.
        diffusion_sqrt = torch.zeros([drift.size(0), drift.size(1), self.state_dim, self.state_dim], device = drift.device) #Create tensor to assign diffusion matrix elements.
        #Assign elements to drift vector.
        drift[:, :, 0 : 1] = drift_SOC
        drift[:, :, 1 : 2] = drift_DOC
        drift[:, :, 2 : 3] = drift_MBC
        if self.DIFFUSION_TYPE == 'C':
            diffusion_sqrt[:, :, 0 : 1, 0] = torch.sqrt(LowerBound.apply(SCON_params_dict_rep['c_SOC'], 1e-8)) #SOC diffusion standard deviation
            diffusion_sqrt[:, :, 1 : 2, 1] = torch.sqrt(LowerBound.apply(SCON_params_dict_rep['c_DOC'], 1e-8)) #DOC diffusion standard deviation
            diffusion_sqrt[:, :, 2 : 3, 2] = torch.sqrt(LowerBound.apply(SCON_params_dict_rep['c_MBC'], 1e-8)) #MBC diffusion standard deviation
            #diffusion_sqrt_single = torch.diag_embed(torch.sqrt(torch.stack([LowerBound.apply(SCON_params_dict['c_SOC'], 1e-8), LowerBound.apply(SCON_params_dict['c_DOC'], 1e-8), LowerBound.apply(SCON_params_dict['c_MBC'], 1e-8)], 1))) #Create single diffusion matrix by diagonalizing constant noise scale parameters. 
            #diffusion_sqrt = diffusion_sqrt_single.unsqueeze(1).expand(-1, T_SPAN_TENSOR.size(1), -1, -1) #Expand diffusion matrices across all paths and across discretized time steps.           
        elif self.DIFFUSION_TYPE == 'SS':
            diffusion_sqrt[:, :, 0 : 1, 0] = torch.sqrt(LowerBound.apply(SOC * SCON_params_dict_rep['s_SOC'], 1e-8)) #SOC diffusion standard deviation
            diffusion_sqrt[:, :, 1 : 2, 1] = torch.sqrt(LowerBound.apply(DOC * SCON_params_dict_rep['s_DOC'], 1e-8)) #DOC diffusion standard deviation
            diffusion_sqrt[:, :, 2 : 3, 2] = torch.sqrt(LowerBound.apply(MBC * SCON_params_dict_rep['s_MBC'], 1e-8)) #MBC diffusion standard deviation
        
        return drift, diffusion_sqrt

    def drift_diffusion_add_CO2(
        self,
        C_PATH: torch.Tensor, 
        SCON_params_dict: DictOfTensors,
        T_SPAN_TENSOR: torch.Tensor, 
        I_S_TENSOR: torch.Tensor, 
        I_D_TENSOR: torch.Tensor, 
        TEMP_TENSOR: torch.Tensor, 
        TEMP_REF: Number,
        ) -> TupleOfTensors:
        '''
        Accepts states x and dictionary of parameter samples.
        Returns SCON drift and diffusion tensors corresponding to state values and parameter samples, along with tensor of states x concatenated with CO2.  
        Expected SCON_params_dict = {'u_M': u_M, 'a_SD': a_SD, 'a_DS': a_DS, 'a_M': a_M, 'a_MSC': a_MSC, 'k_S_ref': k_S_ref, 'k_D_ref': k_D_ref, 'k_M_ref': k_M_ref, 'Ea_S': Ea_S, 'Ea_D': Ea_D, 'Ea_M': Ea_M, '[cs]_SOC': [cs]_SOC, '[cs]_DOC': [cs]_DOC, '[cs]_MBC': [cs]_MBC}
        '''
        #Partition SOC, DOC, MBC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor. 
        SOC, DOC, MBC =  torch.chunk(C_PATH, self.state_dim, -1)
        #Repeat and permute parameter values to match dimension sizes.
        SCON_params_dict_rep = dict((k, v.repeat(1, T_SPAN_TENSOR.size(1), 1).permute(2, 1, 0)) for k, v in SCON_params_dict.items())
        #Initiate tensor with same dims as C_PATH to assign drift.
        drift = torch.empty_like(C_PATH, device = C_PATH.device)
        #Decay parameters are forced by temperature changes.
        k_S = arrhenius_temp_dep(SCON_params_dict_rep['k_S_ref'], TEMP_TENSOR, SCON_params_dict_rep['Ea_S'], TEMP_REF) #Apply vectorized temperature-dependent transformation to k_S_ref.
        k_D = arrhenius_temp_dep(SCON_params_dict_rep['k_D_ref'], TEMP_TENSOR, SCON_params_dict_rep['Ea_D'], TEMP_REF) #Apply vectorized temperature-dependent transformation to k_D_ref.
        k_M = arrhenius_temp_dep(SCON_params_dict_rep['k_M_ref'], TEMP_TENSOR, SCON_params_dict_rep['Ea_M'], TEMP_REF) #Apply vectorized temperature-dependent transformation to k_M_ref.
        #Drift is calculated.
        drift_SOC = I_S_TENSOR + SCON_params_dict_rep['a_DS'] * k_D * DOC + SCON_params_dict_rep['a_M'] * SCON_params_dict_rep['a_MSC'] * k_M * MBC - k_S * SOC
        drift_DOC = I_D_TENSOR + SCON_params_dict_rep['a_SD'] * k_S * SOC + SCON_params_dict_rep['a_M'] * (1 - SCON_params_dict_rep['a_MSC']) * k_M * MBC - (SCON_params_dict_rep['u_M'] + k_D) * DOC
        drift_MBC = SCON_params_dict_rep['u_M'] * DOC - k_M * MBC
        #Diffusion matrix is computed based on diffusion type.
        diffusion_sqrt = torch.zeros([drift.size(0), drift.size(1), self.state_dim, self.state_dim], device = drift.device) #Create tensor to assign diffusion matrix elements.
        #Assign elements to drift vector.
        drift[:, :, 0 : 1] = drift_SOC
        drift[:, :, 1 : 2] = drift_DOC
        drift[:, :, 2 : 3] = drift_MBC
        if self.DIFFUSION_TYPE == 'C':
            diffusion_sqrt[:, :, 0 : 1, 0] = torch.sqrt(LowerBound.apply(SCON_params_dict_rep['c_SOC'], 1e-8)) #SOC diffusion standard deviation
            diffusion_sqrt[:, :, 1 : 2, 1] = torch.sqrt(LowerBound.apply(SCON_params_dict_rep['c_DOC'], 1e-8)) #DOC diffusion standard deviation
            diffusion_sqrt[:, :, 2 : 3, 2] = torch.sqrt(LowerBound.apply(SCON_params_dict_rep['c_MBC'], 1e-8)) #MBC diffusion standard deviation
            #diffusion_sqrt_single = torch.diag_embed(torch.sqrt(torch.stack([LowerBound.apply(SCON_params_dict['c_SOC'], 1e-8), LowerBound.apply(SCON_params_dict['c_DOC'], 1e-8), LowerBound.apply(SCON_params_dict['c_MBC'], 1e-8)], 1))) #Create single diffusion matrix by diagonalizing constant noise scale parameters. 
            #diffusion_sqrt = diffusion_sqrt_single.unsqueeze(1).expand(-1, T_SPAN_TENSOR.size(1), -1, -1) #Expand diffusion matrices across all paths and across discretized time steps.           
        elif self.DIFFUSION_TYPE == 'SS':
            diffusion_sqrt[:, :, 0 : 1, 0] = torch.sqrt(LowerBound.apply(SOC * SCON_params_dict_rep['s_SOC'], 1e-8)) #SOC diffusion standard deviation
            diffusion_sqrt[:, :, 1 : 2, 1] = torch.sqrt(LowerBound.apply(DOC * SCON_params_dict_rep['s_DOC'], 1e-8)) #DOC diffusion standard deviation
            diffusion_sqrt[:, :, 2 : 3, 2] = torch.sqrt(LowerBound.apply(MBC * SCON_params_dict_rep['s_MBC'], 1e-8)) #MBC diffusion standard deviation
        
        #Compute CO2.        
        CO2 = (k_S * SOC * (1 - SCON_params_dict_rep['a_SD'])) + (k_D * DOC * (1 - SCON_params_dict_rep['a_DS'])) + (k_M * MBC * (1 - SCON_params_dict_rep['a_M']))
        #Add CO2 as additional dimension to original x matrix.
        x_add_CO2 = torch.cat([C_PATH, CO2], -1)

        return drift, diffusion_sqrt, x_add_CO2

    def add_CO2(
        self,
        C_PATH: torch.Tensor,
        SCON_params_dict: DictOfTensors,
        TEMP_TENSOR: torch.Tensor, 
        TEMP_REF: Number,
        ) -> TupleOfTensors:
        '''
        Accepts input of states x and dictionary of parameter samples.
        Returns matrix (re-sized from x) that not only includes states, but added CO2 values in expanded third dimension of tensor.
        '''
        #Partition SOC, DOC, MBC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor. 
        SOC, DOC, MBC =  torch.chunk(C_PATH, self.state_dim, -1)
        #Repeat and permute parameter values to match dimension sizes.
        SCON_params_dict_rep = dict((k, v.repeat(1, TEMP_TENSOR.size(1), 1).permute(2, 1, 0)) for k, v in SCON_params_dict.items())
        #Decay parameters are forced by temperature changes.
        k_S = arrhenius_temp_dep(SCON_params_dict_rep['k_S_ref'], TEMP_TENSOR, SCON_params_dict_rep['Ea_S'], TEMP_REF) #Apply vectorized temperature-dependent transformation to k_S_ref.
        k_D = arrhenius_temp_dep(SCON_params_dict_rep['k_D_ref'], TEMP_TENSOR, SCON_params_dict_rep['Ea_D'], TEMP_REF) #Apply vectorized temperature-dependent transformation to k_D_ref.
        k_M = arrhenius_temp_dep(SCON_params_dict_rep['k_M_ref'], TEMP_TENSOR, SCON_params_dict_rep['Ea_M'], TEMP_REF) #Apply vectorized temperature-dependent transformation to k_M_ref.
        #Compute CO2.
        CO2 = (k_S * SOC * (1 - SCON_params_dict_rep['a_SD'])) + (k_D * DOC * (1 - SCON_params_dict_rep['a_DS'])) + (k_M * MBC * (1 - SCON_params_dict_rep['a_M']))
        #Add CO2 as additional dimension to original x matrix.
        x_add_CO2 = torch.cat([C_PATH, CO2], -1)
        
        return x_add_CO2

class SAWB(SBM_SDE):
    '''
    Class contains SAWB SDE drift (alpha) and diffusion (beta) equations.
    Constant (C) and state-scaling (SS) diffusion paramterizations are included. DIFFUSION_TYPE must thereby be specified as 'C' or 'SS'. 
    Other diffusion parameterizations are not included.
    '''
    def __init__(
        self,
        DIFFUSION_TYPE: str
        ):

        if DIFFUSION_TYPE not in {'C', 'SS'}:
            raise NotImplementedError('Other diffusion parameterizations aside from constant (c) or state-scaling (ss) have not been implemented.')

        self.DIFFUSION_TYPE = DIFFUSION_TYPE
        self.state_dim = 4

    def drift_diffusion(
        self,
        C_PATH: torch.Tensor, 
        SAWB_params_dict: DictOfTensors,
        T_SPAN_TENSOR: torch.Tensor, 
        I_S_TENSOR: torch.Tensor, 
        I_D_TENSOR: torch.Tensor, 
        TEMP_TENSOR: torch.Tensor, 
        TEMP_REF: Number,
        ) -> TupleOfTensors:
        '''
        Accepts states x and dictionary of parameter samples.
        Returns SAWB drift and diffusion tensors corresponding to state values and parameter samples.  
        Expected SAWB_params_dict = {'u_Q_ref': u_Q_ref, 'Q': Q, 'a_MSA': a_MSA, 'K_D': K_D, 'K_U': K_U, 'V_D_ref': V_D_ref, 'V_U_ref': V_U_ref, 'Ea_V_D': Ea_V_D, 'Ea_V_U': Ea_V_U, 'r_M': r_M, 'r_E': r_E, 'r_L': r_L, '[cs]_SOC': [cs]_SOC, '[cs]_DOC': [cs]_DOC, '[cs]_MBC': [cs]_MBC, '[cs]_EEC': [cs]_EEC}
        '''
        #Partition SOC, DOC, MBC, EEC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor.
        SOC, DOC, MBC, EEC =  torch.chunk(C_PATH, state_dim, -1)
        #Repeat and permute parameter values to match dimension sizes.
        SAWB_params_dict_rep = dict((k, v.repeat(1, T_SPAN_TENSOR.size(1), 1).permute(2, 1, 0)) for k, v in SAWB_params_dict.items())
        #Initiate tensor with same dims as C_PATH to assign drift.
        drift = torch.empty_like(C_PATH, device = C_PATH.device)
        #Decay parameters are forced by temperature changes.
        u_Q = linear_temp_dep(SAWB_params_dict_rep['u_Q_ref'], TEMP_TENSOR, SAWB_params_dict_rep['Q'], TEMP_REF) #Apply linear temperature-dependence to u_Q.
        V_D = arrhenius_temp_dep(SAWB_params_dict_rep['V_D_ref'], TEMP_TENSOR, SAWB_params_dict_rep['Ea_V_D'], TEMP_REF) #Apply vectorized temperature-dependent transformation to V_D.
        V_U = arrhenius_temp_dep(SAWB_params_dict_rep['V_U_ref'], TEMP_TENSOR, SAWB_params_dict_rep['Ea_V_U'], TEMP_REF) #Apply vectorized temperature-dependent transformation to V_U.
        #Drift is calculated.
        drift_SOC = I_S_TENSOR + SAWB_params_dict_rep['a_MSA'] * SAWB_params_dict_rep['r_M'] * MBC - ((V_D * EEC * SOC) / (SAWB_params_dict_rep['K_D'] + SOC))
        drift_DOC = I_D_TENSOR + (1 - SAWB_params_dict_rep['a_MSA']) * SAWB_params_dict_rep['r_M'] * MBC + ((V_D * EEC * SOC) / (SAWB_params_dict_rep['K_D'] + SOC)) + SAWB_params_dict_rep['r_L'] * EEC - ((V_U * MBC * DOC) / (SAWB_params_dict_rep['K_U'] + DOC))
        drift_MBC = (u_Q * (V_U * MBC * DOC) / (SAWB_params_dict_rep['K_U'] + DOC)) - (SAWB_params_dict_rep['r_M'] + SAWB_params_dict_rep['r_E']) * MBC
        drift_EEC = SAWB_params_dict_rep['r_E'] * MBC - SAWB_params_dict_rep['r_L'] * EEC
        #Assign elements to drift vector.
        drift[:, :, 0 : 1] = drift_SOC
        drift[:, :, 1 : 2] = drift_DOC
        drift[:, :, 2 : 3] = drift_MBC
        drift[:, :, 3 : 4] = drift_EEC
        #Diffusion matrix is computed based on diffusion type.
        diffusion_sqrt = torch.zeros([drift.size(0), drift.size(1), self.state_dim, self.state_dim], device = drift.device) #Create tensor to assign diffusion matrix elements.            
        if self.DIFFUSION_TYPE == 'C':
            diffusion_sqrt[:, :, 0 : 1, 0] = torch.sqrt(LowerBound.apply(SAWB_params_dict_rep['c_SOC'], 1e-8)) #SOC diffusion standard deviation
            diffusion_sqrt[:, :, 1 : 2, 1] = torch.sqrt(LowerBound.apply(SAWB_params_dict_rep['c_DOC'], 1e-8)) #DOC diffusion standard deviation
            diffusion_sqrt[:, :, 2 : 3, 2] = torch.sqrt(LowerBound.apply(SAWB_params_dict_rep['c_MBC'], 1e-8)) #MBC diffusion standard deviation
            diffusion_sqrt[:, :, 3 : 4, 3] = torch.sqrt(LowerBound.apply(SAWB_params_dict_rep['c_EEC'], 1e-8)) #EEC diffusion standard deviation            
            #diffusion_sqrt_single = torch.diag_embed(torch.sqrt(LowerBound.apply(torch.as_tensor([SAWB_params_dict['c_SOC'], SAWB_params_dict['c_DOC'], SAWB_params_dict['c_MBC'], SAWB_params_dict['c_EEC'], SAWB_params_dict['c_CO2']]), 1e-8))) #Create single diffusion matrix by diagonalizing constant noise scale parameters.            
            #diffusion_sqrt = diffusion_sqrt_single.unsqueeze(1).expand(-1, T_SPAN_TENSOR.size(1), -1, -1) #Expand diffusion matrices across all paths and across discretized time steps.
        elif self.DIFFUSION_TYPE == 'SS':
            diffusion_sqrt[:, :, 0 : 1, 0] = torch.sqrt(LowerBound.apply(SOC * SAWB_params_dict_rep['s_SOC'], 1e-8)) #SOC diffusion standard deviation
            diffusion_sqrt[:, :, 1 : 2, 1] = torch.sqrt(LowerBound.apply(DOC * SAWB_params_dict_rep['s_DOC'], 1e-8)) #DOC diffusion standard deviation
            diffusion_sqrt[:, :, 2 : 3, 2] = torch.sqrt(LowerBound.apply(MBC * SAWB_params_dict_rep['s_MBC'], 1e-8)) #MBC diffusion standard deviation
            diffusion_sqrt[:, :, 3 : 4, 3] = torch.sqrt(LowerBound.apply(EEC * SAWB_params_dict_rep['s_EEC'], 1e-8)) #EEC diffusion standard deviation            
        return drift, diffusion_sqrt

    def drift_diffusion_add_CO2(
        self,
        C_PATH: torch.Tensor, 
        SAWB_params_dict: DictOfTensors,
        T_SPAN_TENSOR: torch.Tensor, 
        I_S_TENSOR: torch.Tensor, 
        I_D_TENSOR: torch.Tensor, 
        TEMP_TENSOR: torch.Tensor, 
        TEMP_REF: Number,
        ) -> TupleOfTensors:
        '''
        Accepts states x and dictionary of parameter samples.
        Returns SAWB drift and diffusion tensors corresponding to state values and parameter samples, along with tensor of states x concatenated with CO2.  
        Expected SAWB_params_dict = {'u_Q_ref': u_Q_ref, 'Q': Q, 'a_MSA': a_MSA, 'K_D': K_D, 'K_U': K_U, 'V_D_ref': V_D_ref, 'V_U_ref': V_U_ref, 'Ea_V_D': Ea_V_D, 'Ea_V_U': Ea_V_U, 'r_M': r_M, 'r_E': r_E, 'r_L': r_L, '[cs]_SOC': [cs]_SOC, '[cs]_DOC': [cs]_DOC, '[cs]_MBC': [cs]_MBC, '[cs]_EEC': [cs]_EEC}
        '''
        #Partition SOC, DOC, MBC, EEC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor.
        SOC, DOC, MBC, EEC =  torch.chunk(C_PATH, state_dim, -1)
        #Repeat and permute parameter values to match dimension sizes.
        SAWB_params_dict_rep = dict((k, v.repeat(1, T_SPAN_TENSOR.size(1), 1).permute(2, 1, 0)) for k, v in SAWB_params_dict.items())
        #Initiate tensor with same dims as C_PATH to assign drift.
        drift = torch.empty_like(C_PATH, device = C_PATH.device)
        #Decay parameters are forced by temperature changes.
        u_Q = linear_temp_dep(SAWB_params_dict_rep['u_Q_ref'], TEMP_TENSOR, SAWB_params_dict_rep['Q'], TEMP_REF) #Apply linear temperature-dependence to u_Q.
        V_D = arrhenius_temp_dep(SAWB_params_dict_rep['V_D_ref'], TEMP_TENSOR, SAWB_params_dict_rep['Ea_V_D'], TEMP_REF) #Apply vectorized temperature-dependent transformation to V_D.
        V_U = arrhenius_temp_dep(SAWB_params_dict_rep['V_U_ref'], TEMP_TENSOR, SAWB_params_dict_rep['Ea_V_U'], TEMP_REF) #Apply vectorized temperature-dependent transformation to V_U.
        #Drift is calculated.
        drift_SOC = I_S_TENSOR + SAWB_params_dict_rep['a_MSA'] * SAWB_params_dict_rep['r_M'] * MBC - ((V_D * EEC * SOC) / (SAWB_params_dict_rep['K_D'] + SOC))
        drift_DOC = I_D_TENSOR + (1 - SAWB_params_dict_rep['a_MSA']) * SAWB_params_dict_rep['r_M'] * MBC + ((V_D * EEC * SOC) / (SAWB_params_dict_rep['K_D'] + SOC)) + SAWB_params_dict_rep['r_L'] * EEC - ((V_U * MBC * DOC) / (SAWB_params_dict_rep['K_U'] + DOC))
        drift_MBC = (u_Q * (V_U * MBC * DOC) / (SAWB_params_dict_rep['K_U'] + DOC)) - (SAWB_params_dict_rep['r_M'] + SAWB_params_dict_rep['r_E']) * MBC
        drift_EEC = SAWB_params_dict_rep['r_E'] * MBC - SAWB_params_dict_rep['r_L'] * EEC
        #Assign elements to drift vector.
        drift[:, :, 0 : 1] = drift_SOC
        drift[:, :, 1 : 2] = drift_DOC
        drift[:, :, 2 : 3] = drift_MBC
        drift[:, :, 3 : 4] = drift_EEC
        #Diffusion matrix is computed based on diffusion type.
        diffusion_sqrt = torch.zeros([drift.size(0), drift.size(1), self.state_dim, self.state_dim], device = drift.device) #Create tensor to assign diffusion matrix elements.            
        if self.DIFFUSION_TYPE == 'C':
            diffusion_sqrt[:, :, 0 : 1, 0] = torch.sqrt(LowerBound.apply(SAWB_params_dict_rep['c_SOC'], 1e-8)) #SOC diffusion standard deviation
            diffusion_sqrt[:, :, 1 : 2, 1] = torch.sqrt(LowerBound.apply(SAWB_params_dict_rep['c_DOC'], 1e-8)) #DOC diffusion standard deviation
            diffusion_sqrt[:, :, 2 : 3, 2] = torch.sqrt(LowerBound.apply(SAWB_params_dict_rep['c_MBC'], 1e-8)) #MBC diffusion standard deviation
            diffusion_sqrt[:, :, 3 : 4, 3] = torch.sqrt(LowerBound.apply(SAWB_params_dict_rep['c_EEC'], 1e-8)) #EEC diffusion standard deviation            
            #diffusion_sqrt_single = torch.diag_embed(torch.sqrt(LowerBound.apply(torch.as_tensor([SAWB_params_dict['c_SOC'], SAWB_params_dict['c_DOC'], SAWB_params_dict['c_MBC'], SAWB_params_dict['c_EEC'], SAWB_params_dict['c_CO2']]), 1e-8))) #Create single diffusion matrix by diagonalizing constant noise scale parameters.            
            #diffusion_sqrt = diffusion_sqrt_single.unsqueeze(1).expand(-1, T_SPAN_TENSOR.size(1), -1, -1) #Expand diffusion matrices across all paths and across discretized time steps.
        elif self.DIFFUSION_TYPE == 'SS':
            diffusion_sqrt[:, :, 0 : 1, 0] = torch.sqrt(LowerBound.apply(SOC * SAWB_params_dict_rep['s_SOC'], 1e-8)) #SOC diffusion standard deviation
            diffusion_sqrt[:, :, 1 : 2, 1] = torch.sqrt(LowerBound.apply(DOC * SAWB_params_dict_rep['s_DOC'], 1e-8)) #DOC diffusion standard deviation
            diffusion_sqrt[:, :, 2 : 3, 2] = torch.sqrt(LowerBound.apply(MBC * SAWB_params_dict_rep['s_MBC'], 1e-8)) #MBC diffusion standard deviation
            diffusion_sqrt[:, :, 3 : 4, 3] = torch.sqrt(LowerBound.apply(EEC * SAWB_params_dict_rep['s_EEC'], 1e-8)) #EEC diffusion standard deviation           
        #Compute CO2.
        CO2 = (1 - u_Q) * (V_U * MBC * DOC) / (SAWB_params_dict_rep['K_U'] + DOC)
        #Add CO2 as additional dimension to original x matrix.
        x_add_CO2 = torch.cat([C_PATH, CO2], -1)

        return drift, diffusion_sqrt, x_add_CO2

    def add_CO2(
        self,
        C_PATH: torch.Tensor,
        SAWB_params_dict: DictOfTensors,
        TEMP_TENSOR: torch.Tensor, 
        TEMP_REF: Number,
        ) -> TupleOfTensors:
        '''
        Accepts input of states x and dictionary of parameter samples.
        Returns matrix (re-sized from x) that not only includes states, but added CO2 values in expanded third dimension of tensor.
        '''
        #Partition SOC, DOC, MBC, and EEC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor. 
        SOC, DOC, MBC, EEC =  torch.chunk(C_PATH, self.state_dim, -1)
        #Repeat and permute parameter values to match dimension sizes.
        SAWB_params_dict_rep = dict((k, v.repeat(1, TEMP_TENSOR.size(1), 1).permute(2, 1, 0)) for k, v in SAWB_params_dict.items())
        #Decay parameters are forced by temperature changes.
        u_Q = linear_temp_dep(SAWB_params_dict_rep['u_Q_ref'], TEMP_TENSOR, SAWB_params_dict_rep['Q'], TEMP_REF) #Apply linear temperature-dependence to u_Q.
        V_D = arrhenius_temp_dep(SAWB_params_dict_rep['V_D_ref'], TEMP_TENSOR, SAWB_params_dict_rep['Ea_V_D'], TEMP_REF) #Apply vectorized temperature-dependent transformation to V_D.
        V_U = arrhenius_temp_dep(SAWB_params_dict_rep['V_U_ref'], TEMP_TENSOR, SAWB_params_dict_rep['Ea_V_U'], TEMP_REF) #Apply vectorized temperature-dependent transformation to V_U.
        #Compute CO2.
        CO2 = (1 - u_Q) * (V_U * MBC * DOC) / (SAWB_params_dict_rep['K_U'] + DOC)
        #Add CO2 as additional dimension to original x matrix.
        x_add_CO2 = torch.cat([C_PATH, CO2], -1)
        
        return x_add_CO2

class SAWB_ECA(SBM_SDE):
    '''
    Class contains SAWB-ECA SDE drift (alpha) and diffusion (beta) equations.
    Constant (C) and state-scaling (SS) diffusion paramterizations are included. DIFFUSION_TYPE must thereby be specified as 'C' or 'SS'. 
    Other diffusion parameterizations are not included.
    '''
    def __init__(
        self,
        DIFFUSION_TYPE: str
        ):

        if DIFFUSION_TYPE not in {'C', 'SS'}:
            raise NotImplementedError('Other diffusion parameterizations aside from constant (c) or state-scaling (ss) have not been implemented.')

        self.DIFFUSION_TYPE = DIFFUSION_TYPE
        self.state_dim = 4
    
    def drift_diffusion(
        self,
        C_PATH: torch.Tensor, 
        SAWB_ECA_params_dict: DictOfTensors,
        T_SPAN_TENSOR: torch.Tensor, 
        I_S_TENSOR: torch.Tensor, 
        I_D_TENSOR: torch.Tensor, 
        TEMP_TENSOR: torch.Tensor, 
        TEMP_REF: Number,
        ) -> TupleOfTensors:
        '''
        Accepts states x and dictionary of parameter samples.
        Returns SAWB-ECA drift and diffusion tensors corresponding to state values and parameter samples.  
        Expected SAWB_ECA_params_dict = {'u_Q_ref': u_Q_ref, 'Q': Q, 'a_MSA': a_MSA, 'K_DE': K_DE, 'K_UE': K_UE, 'V_DE_ref': V_DE_ref, 'V_UE_ref': V_UE_ref, 'Ea_V_DE': Ea_V_DE, 'Ea_V_UE': Ea_V_UE, 'r_M': r_M, 'r_E': r_E, 'r_L': r_L, '[cs]_SOC': [cs]_SOC, '[cs]_DOC': [cs]_DOC, '[cs]_MBC': [cs]_MBC, '[cs]_EEC': [cs]_EEC}
        '''
        #Partition SOC, DOC, MBC, EEC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor.
        SOC, DOC, MBC, EEC =  torch.chunk(C_PATH, state_dim, -1)
        #Repeat and permute parameter values to match dimension sizes.
        SAWB_ECA_params_dict_rep = dict((k, v.repeat(1, T_SPAN_TENSOR.size(1), 1).permute(2, 1, 0)) for k, v in SAWB_ECA_params_dict_rep.items())
        #Initiate tensor with same dims as C_PATH to assign drift.
        drift = torch.empty_like(C_PATH, device = C_PATH.device)
        #Decay parameters are forced by temperature changes.
        u_Q = linear_temp_dep(SAWB_ECA_params_dict_rep['u_Q_ref'], TEMP_TENSOR, SAWB_ECA_params_dict_rep['Q'], TEMP_REF) #Apply linear temperature-dependence to u_Q.
        V_DE = arrhenius_temp_dep(SAWB_ECA_params_dict_rep['V_DE_ref'], TEMP_TENSOR, SAWB_ECA_params_dict_rep['Ea_V_DE'], TEMP_REF) #Apply vectorized temperature-dependent transformation to V_DE.
        V_UE = arrhenius_temp_dep(SAWB_ECA_params_dict_rep['V_UE_ref'], TEMP_TENSOR, SAWB_ECA_params_dict_rep['Ea_V_UE'], TEMP_REF) #Apply vectorized temperature-dependent transformation to V_UE.
        #Drift is calculated.
        drift_SOC = I_S_TENSOR + SAWB_ECA_params_dict_rep['a_MSA'] * SAWB_ECA_params_dict_rep['r_M'] * MBC - ((V_DE * EEC * SOC) / (SAWB_ECA_params_dict_rep['K_DE'] + EEC + SOC))
        drift_DOC = I_D_TENSOR + (1 - SAWB_ECA_params_dict_rep['a_MSA']) * SAWB_ECA_params_dict_rep['r_M'] * MBC + ((V_DE * EEC * SOC) / (SAWB_ECA_params_dict_rep['K_DE'] + EEC + SOC)) + SAWB_ECA_params_dict_rep['r_L'] * EEC - ((V_UE * MBC * DOC) / (SAWB_ECA_params_dict_rep['K_UE'] + MBC + DOC))
        drift_MBC = (u_Q * (V_UE * MBC * DOC) / (SAWB_ECA_params_dict_rep['K_UE'] + MBC + DOC)) - (SAWB_ECA_params_dict_rep['r_M'] + SAWB_ECA_params_dict_rep['r_E']) * MBC
        drift_EEC = SAWB_ECA_params_dict_rep['r_E'] * MBC - SAWB_ECA_params_dict_rep['r_L'] * EEC
        #Assign elements to drift vector.
        drift[:, :, 0 : 1] = drift_SOC
        drift[:, :, 1 : 2] = drift_DOC
        drift[:, :, 2 : 3] = drift_MBC
        drift[:, :, 3 : 4] = drift_EEC
        #Diffusion matrix is computed based on diffusion type.
        diffusion_sqrt = torch.zeros([drift.size(0), drift.size(1), self.state_dim, self.state_dim], device = drift.device) #Create tensor to assign diffusion matrix elements.            
        if self.DIFFUSION_TYPE == 'C':
            diffusion_sqrt[:, :, 0 : 1, 0] = torch.sqrt(LowerBound.apply(SAWB_ECA_params_dict_rep['c_SOC'], 1e-8)) #SOC diffusion standard deviation
            diffusion_sqrt[:, :, 1 : 2, 1] = torch.sqrt(LowerBound.apply(SAWB_ECA_params_dict_rep['c_DOC'], 1e-8)) #DOC diffusion standard deviation
            diffusion_sqrt[:, :, 2 : 3, 2] = torch.sqrt(LowerBound.apply(SAWB_ECA_params_dict_rep['c_MBC'], 1e-8)) #MBC diffusion standard deviation
            diffusion_sqrt[:, :, 3 : 4, 3] = torch.sqrt(LowerBound.apply(SAWB_ECA_params_dict_rep['c_EEC'], 1e-8)) #EEC diffusion standard deviation            
            #diffusion_sqrt_single = torch.diag_embed(torch.sqrt(LowerBound.apply(torch.as_tensor([SAWB_ECA_params_dict['c_SOC'], SAWB_ECA_params_dict['c_DOC'], SAWB_ECA_params_dict['c_MBC'], SAWB_ECA_params_dict['c_EEC'], SAWB_ECA_params_dict['c_CO2']]), 1e-8))) #Create single diffusion matrix by diagonalizing constant noise scale parameters.            
            #diffusion_sqrt = diffusion_sqrt_single.unsqueeze(1).expand(-1, T_SPAN_TENSOR.size(1), -1, -1) #Expand diffusion matrices across all paths and across discretized time steps.
        elif self.DIFFUSION_TYPE == 'SS':
            diffusion_sqrt[:, :, 0 : 1, 0] = torch.sqrt(LowerBound.apply(SOC * SAWB_ECA_params_dict_rep['s_SOC'], 1e-8)) #SOC diffusion standard deviation
            diffusion_sqrt[:, :, 1 : 2, 1] = torch.sqrt(LowerBound.apply(DOC * SAWB_ECA_params_dict_rep['s_DOC'], 1e-8)) #DOC diffusion standard deviation
            diffusion_sqrt[:, :, 2 : 3, 2] = torch.sqrt(LowerBound.apply(MBC * SAWB_ECA_params_dict_rep['s_MBC'], 1e-8)) #MBC diffusion standard deviation
            diffusion_sqrt[:, :, 3 : 4, 3] = torch.sqrt(LowerBound.apply(EEC * SAWB_ECA_params_dict_rep['s_EEC'], 1e-8)) #EEC diffusion standard deviation           
        return drift, diffusion_sqrt
        

    def drift_diffusion_add_CO2(
        self,
        C_PATH: torch.Tensor, 
        SAWB_ECA_params_dict: DictOfTensors,
        T_SPAN_TENSOR: torch.Tensor, 
        I_S_TENSOR: torch.Tensor, 
        I_D_TENSOR: torch.Tensor, 
        TEMP_TENSOR: torch.Tensor, 
        TEMP_REF: Number,
        ) -> TupleOfTensors:
        '''
        Accepts states x and dictionary of parameter samples.
        Returns SAWB-ECA drift and diffusion tensors corresponding to state values and parameter samples, along with tensor of states x concatenated with CO2 
        Expected SAWB_ECA_params_dict = {'u_Q_ref': u_Q_ref, 'Q': Q, 'a_MSA': a_MSA, 'K_DE': K_DE, 'K_UE': K_UE, 'V_DE_ref': V_DE_ref, 'V_UE_ref': V_UE_ref, 'Ea_V_DE': Ea_V_DE, 'Ea_V_UE': Ea_V_UE, 'r_M': r_M, 'r_E': r_E, 'r_L': r_L, '[cs]_SOC': [cs]_SOC, '[cs]_DOC': [cs]_DOC, '[cs]_MBC': [cs]_MBC, '[cs]_EEC': [cs]_EEC}
        '''
        #Partition SOC, DOC, MBC, EEC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor.
        SOC, DOC, MBC, EEC =  torch.chunk(C_PATH, state_dim, -1)
        #Repeat and permute parameter values to match dimension sizes.
        SAWB_ECA_params_dict_rep = dict((k, v.repeat(1, T_SPAN_TENSOR.size(1), 1).permute(2, 1, 0)) for k, v in SAWB_ECA_params_dict_rep.items())
        #Initiate tensor with same dims as C_PATH to assign drift.
        drift = torch.empty_like(C_PATH, device = C_PATH.device)
        #Decay parameters are forced by temperature changes.
        u_Q = linear_temp_dep(SAWB_ECA_params_dict_rep['u_Q_ref'], TEMP_TENSOR, SAWB_ECA_params_dict_rep['Q'], TEMP_REF) #Apply linear temperature-dependence to u_Q.
        V_DE = arrhenius_temp_dep(SAWB_ECA_params_dict_rep['V_DE_ref'], TEMP_TENSOR, SAWB_ECA_params_dict_rep['Ea_V_DE'], TEMP_REF) #Apply vectorized temperature-dependent transformation to V_DE.
        V_UE = arrhenius_temp_dep(SAWB_ECA_params_dict_rep['V_UE_ref'], TEMP_TENSOR, SAWB_ECA_params_dict_rep['Ea_V_UE'], TEMP_REF) #Apply vectorized temperature-dependent transformation to V_UE.
        #Drift is calculated.
        drift_SOC = I_S_TENSOR + SAWB_ECA_params_dict_rep['a_MSA'] * SAWB_ECA_params_dict_rep['r_M'] * MBC - ((V_DE * EEC * SOC) / (SAWB_ECA_params_dict_rep['K_DE'] + EEC + SOC))
        drift_DOC = I_D_TENSOR + (1 - SAWB_ECA_params_dict_rep['a_MSA']) * SAWB_ECA_params_dict_rep['r_M'] * MBC + ((V_DE * EEC * SOC) / (SAWB_ECA_params_dict_rep['K_DE'] + EEC + SOC)) + SAWB_ECA_params_dict_rep['r_L'] * EEC - ((V_UE * MBC * DOC) / (SAWB_ECA_params_dict_rep['K_UE'] + MBC + DOC))
        drift_MBC = (u_Q * (V_UE * MBC * DOC) / (SAWB_ECA_params_dict_rep['K_UE'] + MBC + DOC)) - (SAWB_ECA_params_dict_rep['r_M'] + SAWB_ECA_params_dict_rep['r_E']) * MBC
        drift_EEC = SAWB_ECA_params_dict_rep['r_E'] * MBC - SAWB_ECA_params_dict_rep['r_L'] * EEC
        #Assign elements to drift vector.
        drift[:, :, 0 : 1] = drift_SOC
        drift[:, :, 1 : 2] = drift_DOC
        drift[:, :, 2 : 3] = drift_MBC
        drift[:, :, 3 : 4] = drift_EEC
        #Diffusion matrix is computed based on diffusion type.
        diffusion_sqrt = torch.zeros([drift.size(0), drift.size(1), self.state_dim, self.state_dim], device = drift.device) #Create tensor to assign diffusion matrix elements.            
        if self.DIFFUSION_TYPE == 'C':
            diffusion_sqrt[:, :, 0 : 1, 0] = torch.sqrt(LowerBound.apply(SAWB_ECA_params_dict_rep['c_SOC'], 1e-8)) #SOC diffusion standard deviation
            diffusion_sqrt[:, :, 1 : 2, 1] = torch.sqrt(LowerBound.apply(SAWB_ECA_params_dict_rep['c_DOC'], 1e-8)) #DOC diffusion standard deviation
            diffusion_sqrt[:, :, 2 : 3, 2] = torch.sqrt(LowerBound.apply(SAWB_ECA_params_dict_rep['c_MBC'], 1e-8)) #MBC diffusion standard deviation
            diffusion_sqrt[:, :, 3 : 4, 3] = torch.sqrt(LowerBound.apply(SAWB_ECA_params_dict_rep['c_EEC'], 1e-8)) #EEC diffusion standard deviation            
            #diffusion_sqrt_single = torch.diag_embed(torch.sqrt(LowerBound.apply(torch.as_tensor([SAWB_ECA_params_dict['c_SOC'], SAWB_ECA_params_dict['c_DOC'], SAWB_ECA_params_dict['c_MBC'], SAWB_ECA_params_dict['c_EEC'], SAWB_ECA_params_dict['c_CO2']]), 1e-8))) #Create single diffusion matrix by diagonalizing constant noise scale parameters.            
            #diffusion_sqrt = diffusion_sqrt_single.unsqueeze(1).expand(-1, T_SPAN_TENSOR.size(1), -1, -1) #Expand diffusion matrices across all paths and across discretized time steps.
        elif self.DIFFUSION_TYPE == 'SS':
            diffusion_sqrt[:, :, 0 : 1, 0] = torch.sqrt(LowerBound.apply(SOC * SAWB_ECA_params_dict_rep['s_SOC'], 1e-8)) #SOC diffusion standard deviation
            diffusion_sqrt[:, :, 1 : 2, 1] = torch.sqrt(LowerBound.apply(DOC * SAWB_ECA_params_dict_rep['s_DOC'], 1e-8)) #DOC diffusion standard deviation
            diffusion_sqrt[:, :, 2 : 3, 2] = torch.sqrt(LowerBound.apply(MBC * SAWB_ECA_params_dict_rep['s_MBC'], 1e-8)) #MBC diffusion standard deviation
            diffusion_sqrt[:, :, 3 : 4, 3] = torch.sqrt(LowerBound.apply(EEC * SAWB_ECA_params_dict_rep['s_EEC'], 1e-8)) #EEC diffusion standard deviation           
        #Compute CO2.
        CO2 = (1 - u_Q) * (V_UE * MBC * DOC) / (SAWB_ECA_params_dict_rep['K_UE'] + DOC)
        #Add CO2 as additional dimension to original x matrix.
        x_add_CO2 = torch.cat([C_PATH, CO2], -1)

        return drift, diffusion_sqrt, x_add_CO2

    def add_CO2(
        self,
        C_PATH: torch.Tensor,
        SAWB_ECA_params_dict: DictOfTensors,
        TEMP_TENSOR: torch.Tensor, 
        TEMP_REF: Number,
        ) -> TupleOfTensors:
        '''
        Accepts input of states x and dictionary of parameter samples.
        Returns matrix (re-sized from x) that not only includes states, but added CO2 values in expanded third dimension of tensor.
        '''
        #Partition SOC, DOC, MBC, and EEC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor. 
        SOC, DOC, MBC, EEC =  torch.chunk(C_PATH, self.state_dim, -1)
        #Repeat and permute parameter values to match dimension sizes.
        SAWB_ECA_params_dict_rep = dict((k, v.repeat(1, TEMP_TENSOR.size(1), 1).permute(2, 1, 0)) for k, v in SAWB_ECA_params_dict.items())
        #Decay parameters are forced by temperature changes.
        u_Q = linear_temp_dep(SAWB_ECA_params_dict_rep['u_Q_ref'], TEMP_TENSOR, SAWB_ECA_params_dict_rep['Q'], TEMP_REF) #Apply linear temperature-dependence to u_Q.
        V_DE = arrhenius_temp_dep(SAWB_ECA_params_dict_rep['V_DE_ref'], TEMP_TENSOR, SAWB_ECA_params_dict_rep['Ea_V_DE'], TEMP_REF) #Apply vectorized temperature-dependent transformation to V_DE.
        V_UE = arrhenius_temp_dep(SAWB_ECA_params_dict_rep['V_UE_ref'], TEMP_TENSOR, SAWB_ECA_params_dict_rep['Ea_V_UE'], TEMP_REF) #Apply vectorized temperature-dependent transformation to V_UE.
        #Compute CO2.
        CO2 = (1 - u_Q) * (V_UE * MBC * DOC) / (SAWB_ECA_params_dict_rep['K_UE'] + DOC)
        #Add CO2 as additional dimension to original x matrix.
        x_add_CO2 = torch.cat([C_PATH, CO2], -1)
        
        return x_add_CO2
