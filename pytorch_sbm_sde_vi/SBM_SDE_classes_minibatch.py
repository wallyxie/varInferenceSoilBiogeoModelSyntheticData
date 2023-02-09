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

    def __init__(
            self,
            T_SPAN_TENSOR: torch.Tensor,
            I_S_TENSOR: torch.Tensor,
            I_D_TENSOR: torch.Tensor,
            TEMP_TENSOR: torch.Tensor,
            TEMP_REF: Number
            ):
        
        self.times = T_SPAN_TENSOR # (1, N, 1)
        self.i_S = I_S_TENSOR
        self.i_D = I_D_TENSOR
        self.temps = TEMP_TENSOR
        self.temp_ref = TEMP_REF

    def drift_diffusion(
        self,
        C_PATH: torch.Tensor, # (batch_size, minibatch_size, state_dim)
        params_dict: DictOfTensors,
        start_idx=0, end_idx=None, diffusion_matrix=True
        ) -> TupleOfTensors:

        #Appropriately index tensors based on minibatch indices and
        #order of operations in data generating process
        if end_idx is None: end_idx = self.temps.shape[1] # if not provided, set end_idx = N
        c_path_drift_diffusion = C_PATH[:, :-1, :] # (batch_size, minibatch_size-1, state_dim)
        i_S_tensor_drift_diffusion = self.i_S[:, start_idx+1:end_idx, :]
        i_D_tensor_drift_diffusion = self.i_D[:, start_idx+1:end_idx, :]
        temp_tensor_drift_diffusion = self.temps[:, start_idx+1:end_idx, :] # (1, minibatch_size-1, 1)

        #Reshape parameter values to match dimension sizes.
        params_dict_res = dict((k, v.reshape(-1, 1, 1)) for k, v in params_dict.items()) # (batch_size) -> (batch_size, 1, 1)

        # Calculate drift
        drift = self.calc_drift(c_path_drift_diffusion, params_dict_res, i_S_tensor_drift_diffusion,
                                i_D_tensor_drift_diffusion, temp_tensor_drift_diffusion) # (batch_size, N-1, state_dim)

        #Diffusion matrix is computed based on diffusion type.
        diffusion_sqrt = self.calc_diffusion_sqrt(c_path_drift_diffusion, params_dict_res, diffusion_matrix)
        
        return drift, diffusion_sqrt

    def drift_diffusion_multi(
        self,
        C_PATH: torch.Tensor, 
        params_dict: DictOfTensors,
        ) -> TupleOfTensors:

        #Appropriately index tensors based on order of operations in data generating process.
        c_path_drift_diffusion = C_PATH[:, :, :-1, :] # (batch_size, num_seqs, N-1, state_dim)
        #t_span_tensor_drift_diffusion = self.times[:, 1:, :].unsqueeze(1) # (1, 1, N-1, 1)
        i_S_tensor_drift_diffusion = self.i_S[:, 1:, :].unsqueeze(1)
        i_D_tensor_drift_diffusion = self.i_D[:, 1:, :].unsqueeze(1)
        temp_tensor_drift_diffusion = self.temps[:, 1:, :].unsqueeze(1)

        #Reshape parameter values to match dimension sizes.
        params_dict_res = dict((k, v.reshape(-1, 1, 1, 1)) for k, v in params_dict.items()) # (batch_size) -> (batch_size, 1, 1, 1)

        # Calculate drift
        drift = self.calc_drift(c_path_drift_diffusion, params_dict_res, i_S_tensor_drift_diffusion,
                                i_D_tensor_drift_diffusion, temp_tensor_drift_diffusion) # (batch_size, N-1, state_dim)

        #Diffusion matrix is computed based on diffusion type.
        diffusion_sqrt = self.calc_diffusion_sqrt(c_path_drift_diffusion, params_dict_res)
        
        return drift, diffusion_sqrt

    def drift_diffusion_add_CO2(
        self,
        C_PATH: torch.Tensor, # (batch_size, minibatch_size, state_dim)
        params_dict: DictOfTensors,
        start_idx=0, end_idx=None, diffusion_matrix=True
        ) -> TupleOfTensors:
        '''
        Accepts states x and dictionary of parameter samples.
        Returns SCON drift and diffusion tensors corresponding to state values and parameter samples, along with tensor of states x concatenated with CO2.  
        Expected SCON_params_dict = {'u_M': u_M, 'a_SD': a_SD, 'a_DS': a_DS, 'a_M': a_M, 'a_MSC': a_MSC, 'k_S_ref': k_S_ref, 'k_D_ref': k_D_ref, 'k_M_ref': k_M_ref, 'Ea_S': Ea_S, 'Ea_D': Ea_D, 'Ea_M': Ea_M, '[cs]_SOC': [cs]_SOC, '[cs]_DOC': [cs]_DOC, '[cs]_MBC': [cs]_MBC}
        '''
        if end_idx is None: end_idx = self.temps.shape[1]
        c_path_drift_diffusion = C_PATH[:, :-1, :] # (batch_size, minibatch_size-1, state_dim)
        i_S_tensor_drift_diffusion = self.i_S[:, start_idx+1:end_idx, :]
        i_D_tensor_drift_diffusion = self.i_D[:, start_idx+1:end_idx, :]
        temp_tensor = self.temps[:, start_idx:end_idx, :]
        
        #Reshape parameter values to match dimension sizes.
        params_dict_res = dict((k, v.reshape(-1, 1, 1)) for k, v in params_dict.items()) # (batch_size) -> (batch_size, 1, 1)

        # Calculate drift and CO2
        drift, CO2 = self.calc_drift_and_CO2(C_PATH, params_dict_res, i_S_tensor_drift_diffusion,
                                             i_D_tensor_drift_diffusion, temp_tensor)
        
        #Add CO2 as additional dimension to original x matrix.
        x_add_CO2 = torch.cat([C_PATH, CO2], -1)

        #Diffusion matrix is computed based on diffusion type.
        diffusion_sqrt = self.calc_diffusion_sqrt(c_path_drift_diffusion, params_dict_res, diffusion_matrix)
        
        return drift, diffusion_sqrt, x_add_CO2

    def drift_diffusion_add_CO2_multi(
        self,
        C_PATH: torch.Tensor, 
        params_dict: DictOfTensors,
        ) -> TupleOfTensors:

        #Appropriately index tensors based on order of operations in data generating process.
        c_path_drift_diffusion = C_PATH[:, :, :-1, :] # (batch_size, num_seqs, N-1, state_dim)
        i_S_tensor_drift_diffusion = self.i_S[:, 1:, :].unsqueeze(1)
        i_D_tensor_drift_diffusion = self.i_D[:, 1:, :].unsqueeze(1)
        temp_tensor = self.temps.unsqueeze(1)

        #Reshape parameter values to match dimension sizes.
        params_dict_res = dict((k, v.reshape(-1, 1, 1, 1)) for k, v in params_dict.items()) # (batch_size) -> (batch_size, 1, 1, 1)

        # Calculate drift
        drift, CO2 = self.calc_drift_and_CO2(C_PATH, params_dict_res, i_S_tensor_drift_diffusion,
                                             i_D_tensor_drift_diffusion, temp_tensor) # (batch_size, num_seqs, N-1, state_dim)

        #Add CO2 as additional dimension to original x matrix.
        x_add_CO2 = torch.cat([C_PATH, CO2], -1)

        #Diffusion matrix is computed based on diffusion type.
        diffusion_sqrt = self.calc_diffusion_sqrt(c_path_drift_diffusion, params_dict_res)
        
        return drift, diffusion_sqrt, x_add_CO2

    def add_CO2(
        self,
        C_PATH: torch.Tensor,
        params_dict: DictOfTensors,
        start_idx=0, end_idx=None, time_step=1
        ) -> TupleOfTensors:
        '''
        Accepts input of states x and dictionary of parameter samples.
        Returns matrix (re-sized from x) that not only includes states, but added CO2 values in expanded third dimension of tensor.
        '''
        # Appropriately index temp_tensor based on minibatch indices and,
        # if provided, time_step (since CO2 is only computed at observed time steps)
        if end_idx is None: end_idx = self.temps.shape[1]
        temp_tensor = self.temps[:, start_idx:end_idx:time_step, :]

        #Repeat and permute parameter values to match dimension sizes.
        params_dict_res = dict((k, v.reshape(-1, 1, 1)) for k, v in params_dict.items())

        #Compute CO2.
        CO2 = self.calc_CO2(C_PATH, params_dict_res, temp_tensor)
        
        #Add CO2 as additional dimension to original x matrix.
        x_add_CO2 = torch.cat([C_PATH, CO2], -1)
        
        return x_add_CO2
        
class SCON(SBM_SDE):
    '''
    Class contains SCON SDE drift (alpha) and diffusion (beta) equations.
    Constant (C) and state-scaling (SS) diffusion paramterizations are included. DIFFUSION_TYPE must thereby be specified as 'C' or 'SS'. 
    Other diffusion parameterizations are not included.
    '''
    def __init__(
            self,
            T_SPAN_TENSOR: torch.Tensor,
            I_S_TENSOR: torch.Tensor,
            I_D_TENSOR: torch.Tensor,
            TEMP_TENSOR: torch.Tensor,
            TEMP_REF: Number,
            DIFFUSION_TYPE: str
            ):
        super().__init__(T_SPAN_TENSOR, I_S_TENSOR, I_D_TENSOR, TEMP_TENSOR, TEMP_REF)

        if DIFFUSION_TYPE not in {'C', 'SS'}:
            raise NotImplementedError('Other diffusion parameterizations aside from constant (c) or state-scaling (ss) have not been implemented.')

        self.DIFFUSION_TYPE = DIFFUSION_TYPE
        self.state_dim = 3

    def calc_drift(self, x, SCON_params_dict_res, i_S_tensor_drift_diffusion,
                   i_D_tensor_drift_diffusion, temp_tensor_drift_diffusion):
        #Partition SOC, DOC, MBC values. Split based on final x dim, which specifies state variables. 
        SOC, DOC, MBC = torch.chunk(x, self.state_dim, -1)

        #Decay parameters are forced by temperature changes.
        k_S = arrhenius_temp_dep(SCON_params_dict_res['k_S_ref'], temp_tensor_drift_diffusion, SCON_params_dict_res['Ea_S'], self.temp_ref)
        k_D = arrhenius_temp_dep(SCON_params_dict_res['k_D_ref'], temp_tensor_drift_diffusion, SCON_params_dict_res['Ea_D'], self.temp_ref)
        k_M = arrhenius_temp_dep(SCON_params_dict_res['k_M_ref'], temp_tensor_drift_diffusion, SCON_params_dict_res['Ea_M'], self.temp_ref) 

        #Drift is calculated.
        drift_SOC = i_S_tensor_drift_diffusion + SCON_params_dict_res['a_DS'] * k_D * DOC + SCON_params_dict_res['a_M'] * SCON_params_dict_res['a_MSC'] * k_M * MBC - k_S * SOC
        drift_DOC = i_D_tensor_drift_diffusion + SCON_params_dict_res['a_SD'] * k_S * SOC + SCON_params_dict_res['a_M'] * (1 - SCON_params_dict_res['a_MSC']) * k_M * MBC - (SCON_params_dict_res['u_M'] + k_D) * DOC
        drift_MBC = SCON_params_dict_res['u_M'] * DOC - k_M * MBC # (batch_size, N-1, 1)
        drift_list = [drift_SOC, drift_DOC, drift_MBC]

        return torch.cat(drift_list, -1) # (batch_size, N-1, state_dim)

    def calc_drift_and_CO2(self, x, SCON_params_dict_res, i_S_tensor_drift_diffusion,
                           i_D_tensor_drift_diffusion, temp_tensor):
        #Partition SOC, DOC, MBC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor. 
        SOC_full, DOC_full, MBC_full = torch.chunk(x, self.state_dim, -1)
        SOC = SOC_full[..., :-1, :] # (batch_size, minibatch_size - 1, 1) or (batch_size, num_seqs, N-1, 1)
        DOC = DOC_full[..., :-1, :]
        MBC = MBC_full[..., :-1, :]

        #Decay parameters are forced by temperature changes.
        k_S_full = arrhenius_temp_dep(SCON_params_dict_res['k_S_ref'], temp_tensor, SCON_params_dict_res['Ea_S'], self.temp_ref) #Apply vectorized temperature-dependent transformation to k_S_ref.
        k_S = k_S_full[..., 1:, :] # (batch_size, minibatch_size - 1, 1) or or (batch_size, num_seqs, N-1, 1)
        k_D_full = arrhenius_temp_dep(SCON_params_dict_res['k_D_ref'], temp_tensor, SCON_params_dict_res['Ea_D'], self.temp_ref) #Apply vectorized temperature-dependent transformation to k_D_ref.
        k_D = k_D_full[..., 1:, :]
        k_M_full = arrhenius_temp_dep(SCON_params_dict_res['k_M_ref'], temp_tensor, SCON_params_dict_res['Ea_M'], self.temp_ref) #Apply vectorized temperature-dependent transformation to k_M_ref.
        k_M = k_M_full[..., 1:, :]
        
        #Drift is calculated.
        drift_SOC = i_S_tensor_drift_diffusion + SCON_params_dict_res['a_DS'] * k_D * DOC + SCON_params_dict_res['a_M'] * SCON_params_dict_res['a_MSC'] * k_M * MBC - k_S * SOC
        drift_DOC = i_D_tensor_drift_diffusion + SCON_params_dict_res['a_SD'] * k_S * SOC + SCON_params_dict_res['a_M'] * (1 - SCON_params_dict_res['a_MSC']) * k_M * MBC - (SCON_params_dict_res['u_M'] + k_D) * DOC
        drift_MBC = SCON_params_dict_res['u_M'] * DOC - k_M * MBC
        drift_list = [drift_SOC, drift_DOC, drift_MBC]

        #Compute CO2.
        CO2 = (k_S_full * SOC_full * (1 - SCON_params_dict_res['a_SD'])) + (k_D_full * DOC_full * (1 - SCON_params_dict_res['a_DS'])) + (k_M_full * MBC_full * (1 - SCON_params_dict_res['a_M']))

        return torch.cat(drift_list, -1), CO2 # (batch_size, N-1, state_dim), (batch_size, N, 1)

    def calc_diffusion_sqrt(self, x, SCON_params_dict_res, diffusion_matrix=True):
        if self.DIFFUSION_TYPE == 'C':
            diffusion_SOC = SCON_params_dict_res['c_SOC']
            diffusion_DOC = SCON_params_dict_res['c_DOC']
            diffusion_MBC = SCON_params_dict_res['c_MBC']
        elif self.DIFFUSION_TYPE == 'SS':
            SOC, DOC, MBC = torch.chunk(x, self.state_dim, -1)
            diffusion_SOC = SOC * SCON_params_dict_res['s_SOC']
            diffusion_DOC = DOC * SCON_params_dict_res['s_DOC']
            diffusion_MBC = MBC * SCON_params_dict_res['s_MBC']
        diffusion_list = [diffusion_SOC, diffusion_DOC, diffusion_MBC]
        diffusion_sqrt = torch.sqrt(LowerBound.apply(torch.cat(torch.atleast_1d(diffusion_list), -1), 1e-8))
    
        if diffusion_matrix:
            return torch.diag_embed(diffusion_sqrt) # (batch_size, 1 or N-1, state_dim, state_dim)
        else:
            return diffusion_sqrt # (batch_size, 1 or N-1, state_dim)

    def calc_CO2(self, x, SCON_params_dict_res, temp_tensor):
        #Partition SOC, DOC, MBC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor. 
        SOC, DOC, MBC = torch.chunk(x, self.state_dim, -1)
        #print(SOC.shape)  # (batch_size, num_seqs, N-1, state_dim)
        
        #Decay parameters are forced by temperature changes.
        k_S = arrhenius_temp_dep(SCON_params_dict_res['k_S_ref'], temp_tensor, SCON_params_dict_res['Ea_S'], self.temp_ref) #Apply vectorized temperature-dependent transformation to k_S_ref.
        k_D = arrhenius_temp_dep(SCON_params_dict_res['k_D_ref'], temp_tensor, SCON_params_dict_res['Ea_D'], self.temp_ref) #Apply vectorized temperature-dependent transformation to k_D_ref.
        k_M = arrhenius_temp_dep(SCON_params_dict_res['k_M_ref'], temp_tensor, SCON_params_dict_res['Ea_M'], self.temp_ref) #Apply vectorized temperature-dependent transformation to k_M_ref.
        #print(k_S.shape)

        #Compute CO2.
        CO2 = (k_S * SOC * (1 - SCON_params_dict_res['a_SD'])) + (k_D * DOC * (1 - SCON_params_dict_res['a_DS'])) + (k_M * MBC * (1 - SCON_params_dict_res['a_M']))
        
        return CO2

class SAWB(SBM_SDE):
    '''
    Class contains SAWB SDE drift (alpha) and diffusion (beta) equations.
    Constant (C) and state-scaling (SS) diffusion paramterizations are included. DIFFUSION_TYPE must thereby be specified as 'C' or 'SS'. 
    Other diffusion parameterizations are not included.
    '''
    def __init__(
            self,
            T_SPAN_TENSOR: torch.Tensor,
            I_S_TENSOR: torch.Tensor,
            I_D_TENSOR: torch.Tensor,
            TEMP_TENSOR: torch.Tensor,
            TEMP_REF: Number,
            DIFFUSION_TYPE: str
            ):
        super().__init__(T_SPAN_TENSOR, I_S_TENSOR, I_D_TENSOR, TEMP_TENSOR, TEMP_REF)

        if DIFFUSION_TYPE not in {'C', 'SS'}:
            raise NotImplementedError('Other diffusion parameterizations aside from constant (c) or state-scaling (ss) have not been implemented.')

        self.DIFFUSION_TYPE = DIFFUSION_TYPE
        self.state_dim = 4

    def calc_drift(self, x, SAWB_params_dict_res, i_S_tensor_drift_diffusion,
                   i_D_tensor_drift_diffusion, temp_tensor_drift_diffusion):
        SOC, DOC, MBC, EEC = torch.chunk(x, self.state_dim, -1)

        #Decay parameters are forced by temperature changes.
        u_Q = linear_temp_dep(SAWB_params_dict_res['u_Q_ref'], temp_tensor_drift_diffusion, SAWB_params_dict_res['Q'], self.temp_ref) #Apply linear temperature-dependence to u_Q.
        V_D = arrhenius_temp_dep(SAWB_params_dict_res['V_D_ref'], temp_tensor_drift_diffusion, SAWB_params_dict_res['Ea_V_D'], self.temp_ref) #Apply vectorized temperature-dependent transformation to V_D.
        V_U = arrhenius_temp_dep(SAWB_params_dict_res['V_U_ref'], temp_tensor_drift_diffusion, SAWB_params_dict_res['Ea_V_U'], self.temp_ref) #Apply vectorized temperature-dependent transformation to V_U.
        
        #Drift is calculated.
        drift_SOC = i_S_tensor_drift_diffusion + SAWB_params_dict_res['a_MSA'] * SAWB_params_dict_res['r_M'] * MBC - ((V_D * EEC * SOC) / (SAWB_params_dict_res['K_D'] + SOC))
        drift_DOC = i_D_tensor_drift_diffusion + (1 - SAWB_params_dict_res['a_MSA']) * SAWB_params_dict_res['r_M'] * MBC + ((V_D * EEC * SOC) / (SAWB_params_dict_res['K_D'] + SOC)) + SAWB_params_dict_res['r_L'] * EEC - ((V_U * MBC * DOC) / (SAWB_params_dict_res['K_U'] + DOC))
        drift_MBC = (u_Q * (V_U * MBC * DOC) / (SAWB_params_dict_res['K_U'] + DOC)) - (SAWB_params_dict_res['r_M'] + SAWB_params_dict_res['r_E']) * MBC
        drift_EEC = SAWB_params_dict_res['r_E'] * MBC - SAWB_params_dict_res['r_L'] * EEC # (batch_size, N-1, 1)
        drift_list = [drift_SOC, drift_DOC, drift_MBC, drift_EEC]

        return torch.cat(drift_list, -1) # (batch_size, N-1, state_dim)

    def calc_drift_and_CO2(self, x, SAWB_params_dict_res, i_S_tensor_drift_diffusion,
                           i_D_tensor_drift_diffusion, temp_tensor):
        #Partition SOC, DOC, MBC, EEC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor.
        SOC_full, DOC_full, MBC_full, EEC_full = torch.chunk(x, self.state_dim, -1)
        SOC = SOC_full[:, :-1, :]
        DOC = DOC_full[:, :-1, :]
        MBC = MBC_full[:, :-1, :]
        EEC = EEC_full[:, :-1, :]
        
        #Decay parameters are forced by temperature changes.
        u_Q_full = linear_temp_dep(SAWB_params_dict_res['u_Q_ref'], temp_tensor, SAWB_params_dict_res['Q'], self.temp_ref) #Apply linear temperature-dependence to u_Q.
        u_Q = u_Q_full[:, 1:, :]
        V_D_full = arrhenius_temp_dep(SAWB_params_dict_res['V_D_ref'], temp_tensor, SAWB_params_dict_res['Ea_V_D'], self.temp_ref) #Apply vectorized temperature-dependent transformation to V_D.
        V_D = V_D_full[:, 1:, :]
        V_U_full = arrhenius_temp_dep(SAWB_params_dict_res['V_U_ref'], temp_tensor, SAWB_params_dict_res['Ea_V_U'], self.temp_ref) #Apply vectorized temperature-dependent transformation to V_U.
        V_U = V_U_full[:, 1:, :]
        
        #Drift is calculated.
        drift_SOC = i_S_tensor_drift_diffusion + SAWB_params_dict_res['a_MSA'] * SAWB_params_dict_res['r_M'] * MBC - ((V_D * EEC * SOC) / (SAWB_params_dict_res['K_D'] + SOC))
        drift_DOC = i_D_tensor_drift_diffusion + (1 - SAWB_params_dict_res['a_MSA']) * SAWB_params_dict_res['r_M'] * MBC + ((V_D * EEC * SOC) / (SAWB_params_dict_res['K_D'] + SOC)) + SAWB_params_dict_res['r_L'] * EEC - ((V_U * MBC * DOC) / (SAWB_params_dict_res['K_U'] + DOC))
        drift_MBC = (u_Q * (V_U * MBC * DOC) / (SAWB_params_dict_res['K_U'] + DOC)) - (SAWB_params_dict_res['r_M'] + SAWB_params_dict_res['r_E']) * MBC
        drift_EEC = SAWB_params_dict_res['r_E'] * MBC - SAWB_params_dict_res['r_L'] * EEC
        drift_list = [drift_SOC, drift_DOC, drift_MBC, drift_EEC]

        #Compute CO2.
        CO2 = (1 - u_Q_full) * (V_U_full * MBC_full * DOC_full) / (SAWB_params_dict_res['K_U'] + DOC_full)
        
        return torch.cat(drift_list, -1), CO2 # (batch_size, N-1, state_dim), (batch_size, N, 1)

    def calc_diffusion_sqrt(self, x, SAWB_params_dict_res, diffusion_matrix=True):
        if self.DIFFUSION_TYPE == 'C':
            diffusion_SOC = SAWB_params_dict_res['c_SOC']
            diffusion_DOC = SAWB_params_dict_res['c_DOC']
            diffusion_MBC = SAWB_params_dict_res['c_MBC']
            diffusion_EEC = SAWB_params_dict_res['c_EEC']
        elif self.DIFFUSION_TYPE == 'SS':
            SOC, DOC, MBC, EEC = torch.chunk(x, self.state_dim, -1)
            diffusion_SOC = SOC * SAWB_params_dict_res['s_SOC']
            diffusion_DOC = DOC * SAWB_params_dict_res['s_DOC']
            diffusion_MBC = MBC * SAWB_params_dict_res['s_MBC']
            diffusion_EEC = EEC * SAWB_params_dict_res['s_EEC']
        diffusion_list = [diffusion_SOC, diffusion_DOC, diffusion_MBC, diffusion_EEC]
        diffusion_sqrt = torch.sqrt(LowerBound.apply(torch.cat(torch.atleast_1d(diffusion_list), -1), 1e-8))
    
        if diffusion_matrix:
            return torch.diag_embed(diffusion_sqrt) # (batch_size, 1 or N-1, state_dim, state_dim)
        else:
            return diffusion_sqrt

    def calc_CO2(self, x, SAWB_params_dict_res, temp_tensor):
        #Partition SOC, DOC, MBC, and EEC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor. 
        SOC, DOC, MBC, EEC = torch.chunk(x, self.state_dim, -1)
        
        #Decay parameters are forced by temperature changes.
        u_Q = linear_temp_dep(SAWB_params_dict_res['u_Q_ref'], temp_tensor, SAWB_params_dict_res['Q'], self.temp_ref) #Apply linear temperature-dependence to u_Q.
        V_D = arrhenius_temp_dep(SAWB_params_dict_res['V_D_ref'], temp_tensor, SAWB_params_dict_res['Ea_V_D'], self.temp_ref) #Apply vectorized temperature-dependent transformation to V_D.
        V_U = arrhenius_temp_dep(SAWB_params_dict_res['V_U_ref'], temp_tensor, SAWB_params_dict_res['Ea_V_U'], self.temp_ref) #Apply vectorized temperature-dependent transformation to V_U.
        
        #Compute CO2.
        CO2 = (1 - u_Q) * (V_U * MBC * DOC) / (SAWB_params_dict_res['K_U'] + DOC)
        
        return CO2

class SAWB_ECA(SBM_SDE):
    '''
    Class contains SAWB-ECA SDE drift (alpha) and diffusion (beta) equations.
    Constant (C) and state-scaling (SS) diffusion paramterizations are included. DIFFUSION_TYPE must thereby be specified as 'C' or 'SS'. 
    Other diffusion parameterizations are not included.
    '''
    def __init__(
            self,
            T_SPAN_TENSOR: torch.Tensor,
            I_S_TENSOR: torch.Tensor,
            I_D_TENSOR: torch.Tensor,
            TEMP_TENSOR: torch.Tensor,
            TEMP_REF: Number,
            DIFFUSION_TYPE: str
            ):
        super().__init__(T_SPAN_TENSOR, I_S_TENSOR, I_D_TENSOR, TEMP_TENSOR, TEMP_REF)

        if DIFFUSION_TYPE not in {'C', 'SS'}:
            raise NotImplementedError('Other diffusion parameterizations aside from constant (c) or state-scaling (ss) have not been implemented.')

        self.DIFFUSION_TYPE = DIFFUSION_TYPE
        self.state_dim = 4
    
    def calc_drift(self, x, SAWB_ECA_params_dict_res, i_S_tensor_drift_diffusion,
                   i_D_tensor_drift_diffusion, temp_tensor_drift_diffusion):
        #Partition SOC, DOC, MBC, EEC values. Split based on final c_path_drift_diffusion dim, which specifies state variables and is also indexed as dim #2 in tensor.
        SOC, DOC, MBC, EEC = torch.chunk(x, self.state_dim, -1)
        
        #Decay parameters are forced by temperature changes.
        u_Q = linear_temp_dep(SAWB_ECA_params_dict_res['u_Q_ref'], temp_tensor_drift_diffusion, SAWB_ECA_params_dict_res['Q'], self.temp_ref) #Apply linear temperature-dependence to u_Q.
        V_DE = arrhenius_temp_dep(SAWB_ECA_params_dict_res['V_DE_ref'], temp_tensor_drift_diffusion, SAWB_ECA_params_dict_res['Ea_V_DE'], self.temp_ref) #Apply vectorized temperature-dependent transformation to V_DE.
        V_UE = arrhenius_temp_dep(SAWB_ECA_params_dict_res['V_UE_ref'], temp_tensor_drift_diffusion, SAWB_ECA_params_dict_res['Ea_V_UE'], self.temp_ref) #Apply vectorized temperature-dependent transformation to V_UE.
        
        #Drift is calculated.
        drift_SOC = i_S_tensor_drift_diffusion + SAWB_ECA_params_dict_res['a_MSA'] * SAWB_ECA_params_dict_res['r_M'] * MBC - ((V_DE * EEC * SOC) / (SAWB_ECA_params_dict_res['K_DE'] + EEC + SOC))
        drift_DOC = i_D_tensor_drift_diffusion + (1 - SAWB_ECA_params_dict_res['a_MSA']) * SAWB_ECA_params_dict_res['r_M'] * MBC + ((V_DE * EEC * SOC) / (SAWB_ECA_params_dict_res['K_DE'] + EEC + SOC)) + SAWB_ECA_params_dict_res['r_L'] * EEC - ((V_UE * MBC * DOC) / (SAWB_ECA_params_dict_res['K_UE'] + MBC + DOC))
        drift_MBC = (u_Q * (V_UE * MBC * DOC) / (SAWB_ECA_params_dict_res['K_UE'] + MBC + DOC)) - (SAWB_ECA_params_dict_res['r_M'] + SAWB_ECA_params_dict_res['r_E']) * MBC
        drift_EEC = SAWB_ECA_params_dict_res['r_E'] * MBC - SAWB_ECA_params_dict_res['r_L'] * EEC
        drift_list = [drift_SOC, drift_DOC, drift_MBC, drift_EEC]

        return torch.cat(drift_list, -1) # (batch_size, N-1, state_dim)

    def calc_drift_and_CO2(self, x, SAWB_ECA_params_dict_res, i_S_tensor_drift_diffusion,
                           i_D_tensor_drift_diffusion, temp_tensor):
        #Partition SOC, DOC, MBC, EEC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor.
        SOC_full, DOC_full, MBC_full, EEC_full = torch.chunk(x, self.state_dim, -1)
        SOC = SOC_full[..., :-1, :]
        DOC = DOC_full[..., :-1, :]
        MBC = MBC_full[..., :-1, :]
        EEC = EEC_full[..., :-1, :]
        
        #Decay parameters are forced by temperature changes.
        u_Q_full = linear_temp_dep(SAWB_ECA_params_dict_res['u_Q_ref'], temp_tensor, SAWB_ECA_params_dict_res['Q'], self.temp_ref) #Apply linear temperature-dependence to u_Q.
        u_Q = u_Q_full[..., 1:, :]
        V_DE_full = arrhenius_temp_dep(SAWB_ECA_params_dict_res['V_DE_ref'], temp_tensor, SAWB_ECA_params_dict_res['Ea_V_DE'], self.temp_ref) #Apply vectorized temperature-dependent transformation to V_DE.
        V_DE = V_DE_full[..., 1:, :]
        V_UE_full = arrhenius_temp_dep(SAWB_ECA_params_dict_res['V_UE_ref'], temp_tensor, SAWB_ECA_params_dict_res['Ea_V_UE'], self.temp_ref) #Apply vectorized temperature-dependent transformation to V_UE.
        V_UE = V_UE_full[..., 1:, :]
        
        #Drift is calculated.
        drift_SOC = i_S_tensor_drift_diffusion + SAWB_ECA_params_dict_res['a_MSA'] * SAWB_ECA_params_dict_res['r_M'] * MBC - ((V_DE * EEC * SOC) / (SAWB_ECA_params_dict_res['K_DE'] + EEC + SOC))
        drift_DOC = i_D_tensor_drift_diffusion + (1 - SAWB_ECA_params_dict_res['a_MSA']) * SAWB_ECA_params_dict_res['r_M'] * MBC + ((V_DE * EEC * SOC) / (SAWB_ECA_params_dict_res['K_DE'] + EEC + SOC)) + SAWB_ECA_params_dict_res['r_L'] * EEC - ((V_UE * MBC * DOC) / (SAWB_ECA_params_dict_res['K_UE'] + MBC + DOC))
        drift_MBC = (u_Q * (V_UE * MBC * DOC) / (SAWB_ECA_params_dict_res['K_UE'] + MBC + DOC)) - (SAWB_ECA_params_dict_res['r_M'] + SAWB_ECA_params_dict_res['r_E']) * MBC
        drift_EEC = SAWB_ECA_params_dict_res['r_E'] * MBC - SAWB_ECA_params_dict_res['r_L'] * EEC
        drift_list = [drift_SOC, drift_DOC, drift_MBC, drift_EEC]

        #Compute CO2.
        CO2 = (1 - u_Q_full) * (V_UE_full * MBC_full * DOC_full) / (SAWB_ECA_params_dict_res['K_UE'] + MBC_full + DOC_full)

        return torch.cat(drift_list, -1), CO2 # (batch_size, N-1, state_dim), (batch_size, N, 1)

    def calc_diffusion_sqrt(self, x, SAWB_ECA_params_dict_res, diffusion_matrix=True):
        if self.DIFFUSION_TYPE == 'C':
            diffusion_SOC = SAWB_ECA_params_dict_res['c_SOC']
            diffusion_DOC = SAWB_ECA_params_dict_res['c_DOC']
            diffusion_MBC = SAWB_ECA_params_dict_res['c_MBC']
            diffusion_EEC = SAWB_ECA_params_dict_res['c_EEC']
        elif self.DIFFUSION_TYPE == 'SS':
            SOC, DOC, MBC, EEC = torch.chunk(x, self.state_dim, -1)
            diffusion_SOC = SOC * SAWB_ECA_params_dict_res['s_SOC']
            diffusion_DOC = DOC * SAWB_ECA_params_dict_res['s_DOC']
            diffusion_MBC = MBC * SAWB_ECA_params_dict_res['s_MBC']
            diffusion_EEC = EEC * SAWB_ECA_params_dict_res['s_EEC']
        diffusion_list = [diffusion_SOC, diffusion_DOC, diffusion_MBC, diffusion_EEC]
        diffusion_sqrt = torch.sqrt(LowerBound.apply(torch.cat(torch.atleast_1d(diffusion_list), -1), 1e-8))
    
        if diffusion_matrix:
            return torch.diag_embed(diffusion_sqrt) # (batch_size, 1 or N-1, state_dim, state_dim)
        else:
            return diffusion_sqrt # (batch_size, 1 or N-1, state_dim)

    def calc_CO2(self, x, SAWB_ECA_params_dict_res, temp_tensor):
        #Partition SOC, DOC, MBC, and EEC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor. 
        SOC, DOC, MBC, EEC = torch.chunk(x, self.state_dim, -1)
        
        #Decay parameters are forced by temperature changes.
        u_Q = linear_temp_dep(SAWB_ECA_params_dict_res['u_Q_ref'], temp_tensor, SAWB_ECA_params_dict_res['Q'], self.temp_ref) #Apply linear temperature-dependence to u_Q.
        V_DE = arrhenius_temp_dep(SAWB_ECA_params_dict_res['V_DE_ref'], temp_tensor, SAWB_ECA_params_dict_res['Ea_V_DE'], self.temp_ref) #Apply vectorized temperature-dependent transformation to V_DE.
        V_UE = arrhenius_temp_dep(SAWB_ECA_params_dict_res['V_UE_ref'], temp_tensor, SAWB_ECA_params_dict_res['Ea_V_UE'], self.temp_ref) #Apply vectorized temperature-dependent transformation to V_UE.
        
        #Compute CO2.
        CO2 = (1 - u_Q) * (V_UE * MBC * DOC) / (SAWB_ECA_params_dict_res['K_UE'] + MBC + DOC)
        
        return CO2
