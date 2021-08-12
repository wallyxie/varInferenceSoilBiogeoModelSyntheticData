import torch
import numpy as np
from obs_and_flow import LowerBound

'''
This script includes the linear and Arrhenius temperature dependence functions to induce temperature-based forcing in differential equation soil biogeochemical models (SBMs) and the SDE equation functions corresponding to the various parameterizations of the stochastic conventional (SCON), stochastic AWB (SAWB), and stochastic AWB-equilibrium chemistry approximation (SAWB-ECA) for incorporation with normalizing flow "neural stochastic differential equation" solvers. The following SBM SDE system parameterizations are contained in this script:
    1) SCON constant diffusion (SCON-c)
    2) SCON state scaling diffusion (SCON-ss)
    3) SAWB constant diffusion (SAWB-c)
    4) SAWB state scaling diffusion (SAWB-ss)
    5) SAWB-ECA constant diffusion (SAWB-ECA-c)
    6) SAWB-ECA state scaling diffusion (SAWB-ECA-ss)
The respective analytical steady state estimation functions derived from the deterministic ODE versions of the stochastic SBMs are also included in this script.
'''

############################################################
##SOIL BIOGEOCHEMICAL MODEL TEMPERATURE RESPONSE FUNCTIONS##
############################################################

def temp_gen(t, TEMP_REF, TEMP_RISE = 5):
    '''
    This is the temperature forcing function.
    '''
    temp = TEMP_REF + (TEMP_RISE * t) / (80 * 24 * 365) + 10 * torch.sin((2 * np.pi / 24) * t) + 10 * torch.sin((2 * np.pi / (24 * 365)) * t)
    return temp

def arrhenius_temp_dep(parameter, temp, Ea, TEMP_REF):
    '''
    For a parameter with Arrhenius temperature dependence, returns the transformed parameter value.
    0.008314 is the gas constant. Temperatures are in K.
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

##########################################################################
##DETERMINISTIC SOIL BIOGEOCHEMICAL MODEL INITIAL STEADY STATE ESTIMATES##
##########################################################################

def analytical_steady_state_init_CON(SOC_input, DOC_input, SCON_params_dict):
    '''
    Returns a vector of C pool values to initialize an SCON system corresponding to set of parameter values using the analytical steady state equations of the deterministic CON system.
    Vector elements are in order of S_0, D_0, M_0.
    Expected SCON_params_dict = {'u_M': u_M, 'a_SD': a_SD, 'a_DS': a_DS, 'a_M': a_M, 'a_MSC': a_MSC, 'k_S_ref': k_S_ref, 'k_D_ref': k_D_ref, 'k_M_ref': k_M_ref, 'Ea_S': Ea_S, 'Ea_D': Ea_D, 'Ea_M': Ea_M, '[cs]_SOC': [cs]_SOC, '[cs]_DOC': [cs]_DOC, '[cs]_MBC': [cs]_MBC}
    '''
    D_0 = (DOC_input + SOC_input * SCON_params_dict['a_SD']) / (SCON_params_dict['u_M'] + SCON_params_dict['k_D_ref'] + SCON_params_dict['u_M'] * SCON_params_dict['a_M'] * (SCON_params_dict['a_MSC'] - 1 - SCON_params_dict['a_MSC'] * SCON_params_dict['a_SD']) - SCON_params_dict['a_DS'] * SCON_params_dict['k_D_ref'] * SCON_params_dict['a_SD'])
    S_0 = (SOC_input + D_0 * (SCON_params_dict['a_DS'] * SCON_params_dict['k_D_ref'] + SCON_params_dict['u_M'] * SCON_params_dict['a_M'] * SCON_params_dict['a_MSC'])) / SCON_params_dict['k_S_ref']
    M_0 = SCON_params_dict['u_M'] * D_0 / SCON_params_dict['k_M_ref']
    C_0 = torch.stack([S_0, D_0, M_0], 1)
    #C_0_vector = torch.as_tensor([S_0, D_0, M_0]) #Commented out, now that SCON_params_dict is dictionary of tensors.   
    #CO2_0 = SCON_params_dict['k_S_ref'] * S_0 * (1 - SCON_params_dict['a_SD']) + SCON_params_dict['k_D_ref'] * D_0 * (1 - SCON_params_dict['a_DS']) + SCON_params_dict['k_M_ref'] * M_0 * (1 - SCON_params_dict['a_M'])
    #C_0_vector = torch.as_tensor([S_0, D_0, M_0, CO2_0])
    return C_0

def analytical_steady_state_init_AWB(SOC_input, DOC_input, SAWB_params_dict):
    '''
    Returns a vector of C pool values to initialize an SAWB system corresponding to set of parameter values using the analytical steady state equations of the deterministic CON system.
    Vector elements are in order of S_0, D_0, M_0, E_0.
    Expected SAWB_params_dict = {'u_Q_ref': u_Q_ref, 'Q': Q, 'a_MSA': a_MSA, 'K_D': K_D, 'K_U': K_U, 'V_D_ref': V_D_ref, 'V_U_ref': V_U_ref, 'Ea_V_D': Ea_V_D, 'Ea_V_U': Ea_V_U, 'r_M': r_M, 'r_E': r_E, 'r_L': r_L, '[cs]_SOC': [cs]_SOC, '[cs]_DOC': [cs]_DOC, '[cs]_MBC': [cs]_MBC, '[cs]_EEC': [cs]_EEC}
    '''
    S_0 = -((SAWB_params_dict['K_D'] * SAWB_params_dict['r_L'] * (SOC_input * SAWB_params_dict['r_E'] * (SAWB_params_dict['u_Q_ref'] - 1) - SAWB_params_dict['a_MSA'] * DOC_input * SAWB_params_dict['r_M'] * SAWB_params_dict['u_Q_ref'] + SOC_input * SAWB_params_dict['r_M'] * (-1 + SAWB_params_dict['u_Q_ref'] - SAWB_params_dict['a_MSA'] * SAWB_params_dict['u_Q_ref']))) / (DOC_input * SAWB_params_dict['u_Q_ref'] * (-SAWB_params_dict['a_MSA'] * SAWB_params_dict['r_L'] * SAWB_params_dict['r_M'] + SAWB_params_dict['r_E'] * SAWB_params_dict['V_D_ref']) + SOC_input * (SAWB_params_dict['r_E'] * SAWB_params_dict['r_L'] * (SAWB_params_dict['u_Q_ref'] - 1) + SAWB_params_dict['r_L'] * SAWB_params_dict['r_M'] * (-1 + SAWB_params_dict['u_Q_ref'] - SAWB_params_dict['a_MSA'] * SAWB_params_dict['u_Q_ref']) + SAWB_params_dict['r_E'] * SAWB_params_dict['u_Q_ref'] * SAWB_params_dict['V_D_ref'])))
    D_0 = -((SAWB_params_dict['K_U'] * (SAWB_params_dict['r_E'] + SAWB_params_dict['r_M'])) / (SAWB_params_dict['r_E'] + SAWB_params_dict['r_M'] - SAWB_params_dict['u_Q_ref'] * SAWB_params_dict['V_U_ref']))
    M_0 = -((SOC_input + DOC_input) * SAWB_params_dict['u_Q_ref']) / ((SAWB_params_dict['r_E'] + SAWB_params_dict['r_M']) * (SAWB_params_dict['u_Q_ref'] - 1))
    E_0 = SAWB_params_dict['r_E'] * M_0 / SAWB_params_dict['r_L']
    C_0 = torch.stack([S_0, D_0, M_0, E_0], 1)
    #C_0_vector = torch.as_tensor([S_0, D_0, M_0, E_0]) #Commented out, now that SCON_params_dict is dictionary of tensors.
    #E_0 = -((SAWB_params_dict['r_E'] * SAWB_params_dict['u_Q_ref'] * (SOC_input + DOC_input)) / (SAWB_params_dict['r_L'] * (SAWB_params_dict['r_E'] + SAWB_params_dict['r_M']) * (SAWB_params_dict['u_Q_ref'] - 1)))
    #CO2_0 = (1 - SAWB_params_dict['u_Q_ref']) * (SAWB_params_dict['V_U_ref'] * M_0 * D_0) / (SAWB_params_dict['K_U'] + D_0)
    #C_0_vector = torch.as_tensor([S_0, D_0, M_0, E_0, CO2_0])
    return C_0

def analytical_steady_state_init_AWB_ECA(SOC_input, DOC_input, SAWB_ECA_params_dict):
    '''
    Returns a vector of C pool values to initialize an SAWB-ECA system corresponding to set of parameter values using the analytical steady state equations of the deterministic CON system.
    Vector elements are in order of S_0, D_0, M_0, E_0.
    Expected SAWB_ECA_params_dict = {'u_Q_ref': u_Q_ref, 'Q': Q, 'a_MSA': a_MSA, 'K_DE': K_DE, 'K_UE': K_UE, 'V_DE_ref': V_DE_ref, 'V_UE_ref': V_UE_ref, 'Ea_V_DE': Ea_V_DE, 'Ea_V_UE': Ea_V_UE, 'r_M': r_M, 'r_E': r_E, 'r_L': r_L, '[cs]_SOC': [cs]_SOC, '[cs]_DOC': [cs]_DOC, '[cs]_MBC': [cs]_MBC, '[cs]_EEC': [cs]_EEC}
    '''
    S_0 = ((-SAWB_ECA_params_dict['K_DE'] * SAWB_ECA_params_dict['r_L'] * (SAWB_ECA_params_dict['r_E'] + SAWB_ECA_params_dict['r_M']) * (SAWB_ECA_params_dict['u_Q_ref'] - 1) + SAWB_ECA_params_dict['r_E'] * SAWB_ECA_params_dict['u_Q_ref'] * (SOC_input + DOC_input)) * (SOC_input * SAWB_ECA_params_dict['r_E'] * (SAWB_ECA_params_dict['u_Q_ref'] - 1) - SAWB_ECA_params_dict['a_MSA'] * DOC_input * SAWB_ECA_params_dict['r_M'] * SAWB_ECA_params_dict['u_Q_ref'] + SOC_input * SAWB_ECA_params_dict['r_M'] * (SAWB_ECA_params_dict['u_Q_ref'] - SAWB_ECA_params_dict['a_MSA'] * SAWB_ECA_params_dict['u_Q_ref'] - 1))) / ((SAWB_ECA_params_dict['r_E'] + SAWB_ECA_params_dict['r_M']) * (SAWB_ECA_params_dict['u_Q_ref'] - 1) * (DOC_input * SAWB_ECA_params_dict['u_Q_ref'] * (SAWB_ECA_params_dict['r_E'] * SAWB_ECA_params_dict['V_DE_ref'] - SAWB_ECA_params_dict['a_MSA'] * SAWB_ECA_params_dict['r_L'] * SAWB_ECA_params_dict['r_M']) + SOC_input * (SAWB_ECA_params_dict['r_E'] * SAWB_ECA_params_dict['r_L'] * (SAWB_ECA_params_dict['u_Q_ref'] - 1) + SAWB_ECA_params_dict['r_L'] * SAWB_ECA_params_dict['r_M'] * (SAWB_ECA_params_dict['u_Q_ref'] - SAWB_ECA_params_dict['a_MSA'] * SAWB_ECA_params_dict['u_Q_ref'] - 1) + SAWB_ECA_params_dict['r_E'] * SAWB_ECA_params_dict['u_Q_ref'] * SAWB_ECA_params_dict['V_DE_ref'])))
    D_0 = -(SAWB_ECA_params_dict['K_UE'] * (SAWB_ECA_params_dict['r_E'] + SAWB_ECA_params_dict['r_M']) * (SAWB_ECA_params_dict['u_Q_ref'] - 1) - (SOC_input + DOC_input) * SAWB_ECA_params_dict['u_Q_ref']) / ((SAWB_ECA_params_dict['u_Q_ref'] - 1) * (SAWB_ECA_params_dict['r_E'] + SAWB_ECA_params_dict['r_M'] - SAWB_ECA_params_dict['u_Q_ref'] * SAWB_ECA_params_dict['V_UE_ref']))
    M_0 = -((SOC_input + DOC_input) * SAWB_ECA_params_dict['u_Q_ref']) / ((SAWB_ECA_params_dict['r_E'] + SAWB_ECA_params_dict['r_M']) * (SAWB_ECA_params_dict['u_Q_ref'] - 1))
    E_0 = SAWB_ECA_params_dict['r_E'] * M_0 / SAWB_ECA_params_dict['r_L']
    C_0 = torch.stack([S_0, D_0, M_0, E_0], 1)    
    #C_0_vector = torch.as_tensor([S_0, D_0, M_0, E_0]) #Commented out, now that SCON_params_dict is dictionary of tensors.    
    #E_0 = -((SAWB_params_dict['r_E'] * SAWB_params_dict['u_Q_ref'] * (SOC_input + DOC_input)) / (SAWB_params_dict['r_L'] * (SAWB_params_dict['r_E'] + SAWB_params_dict['r_M']) * (SAWB_params_dict['u_Q_ref'] - 1)))
    #CO2_0 = (1 - SAWB_ECA_params_dict['u_Q_ref']) * SAWB_ECA_params_dict['V_UE_ref'] * M_0 * D_0 / (SAWB_ECA_params_dict['K_UE'] + M_0 + D_0)
    #C_0_vector = torch.as_tensor([S_0, D_0, M_0, E_0, CO2_0])
    return C_0

####################################################
##STOCHASTIC DIFFERENTIAL EQUATION MODEL FUNCTIONS##
####################################################

#SCONR-C fix u_M, a, and Ea experiment
def drift_diffusion_SCONR_C_fix_a_Ea(C_PATH, T_SPAN_TENSOR, I_S_TENSOR, I_D_TENSOR, TEMP_TENSOR, TEMP_REF, SCONR_C_fix_u_M_a_Ea_params_dict):
    '''
    Returns SCONR fix u_M, a, and Ea "constant diffusion parameterization" drift vectors and diffusion matrices.
    Expected SCONR_C_fix_u_M_a_Ea_params_dict = {'k_S_ref': k_S_ref, 'k_D_ref': k_D_ref, 'k_M_ref': k_M_ref, 'c_SOC': c_SOC, 'c_DOC': c_DOC, 'c_MBC': c_MBC}
    '''
    state_dim = 3 #SCON has three state variables in SOC, DOC, and MBC.
    SOC, DOC, MBC =  torch.chunk(C_PATH, state_dim, -1) #Partition SOC, DOC, and MBC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor. 
    drift = torch.empty_like(C_PATH, device = C_PATH.device) #Initiate tensor with same dims as C_PATH to assign drift.
    #Decay parameters are forced by temperature changes.
    k_S = arrhenius_temp_dep(SCONR_C_fix_u_M_a_Ea_params_dict['k_S_ref'], TEMP_TENSOR, 55, TEMP_REF) #Apply vectorized temperature-dependent transformation to k_S_ref.
    k_S = k_S.permute(2, 1, 0) #Get k_S into appropriate dimensions. 
    k_D = arrhenius_temp_dep(SCONR_C_fix_u_M_a_Ea_params_dict['k_D_ref'], TEMP_TENSOR, 48, TEMP_REF) #Apply vectorized temperature-dependent transformation to k_D_ref.
    k_D = k_D.permute(2, 1, 0) #Get k_D into appropriate dimensions.
    k_M = arrhenius_temp_dep(SCONR_C_fix_u_M_a_Ea_params_dict['k_M_ref'], TEMP_TENSOR, 48, TEMP_REF) #Apply vectorized temperature-dependent transformation to k_M_ref.
    k_M = k_M.permute(2, 1, 0) #Get k_M into appropriate dimensions.
    #Repeat and permute parameter values to match dimension sizes
    SCONR_C_fix_u_M_a_Ea_params_dict_rep = dict((k, v.repeat(1, T_SPAN_TENSOR.size(1), 1).permute(2, 1, 0)) for k, v in SCONR_C_fix_u_M_a_Ea_params_dict.items())    
    #Drift is calculated.
    drift_SOC = I_S_TENSOR + 0.5 * k_D * DOC + 0.5 * k_M * MBC - k_S * SOC
    drift_DOC = I_D_TENSOR + 0.5 * k_S * SOC + 0.5 * k_M * MBC - (0.0016 + k_D) * DOC
    drift_MBC = 0.0016 * DOC - k_M * MBC
    #CO2 = (k_S * SOC * (1 - SCONR_C_fix_u_M_a_Ea_params_dict_rep['a_SD'])) + (k_D * DOC * (1 - SCONR_C_fix_u_M_a_Ea_params_dict_rep['a_DS'])) + (k_M * MBC * (1 - SCONR_C_fix_u_M_a_Ea_params_dict_rep['a_M'])) 
    #Assign elements to drift vector.
    drift[:, :, 0 : 1] = drift_SOC
    drift[:, :, 1 : 2] = drift_DOC
    drift[:, :, 2 : 3] = drift_MBC
    #drift[:, :, 3 : 4] = CO2 #CO2 is not a part of the drift. This is a hack for the explicit algebraic variable situation.
    #Diffusion matrix is assigned.
    diffusion_sqrt_single = torch.diag_embed(torch.sqrt(torch.stack([LowerBound.apply(SCONR_C_fix_u_M_a_Ea_params_dict['c_SOC'], 1e-8), LowerBound.apply(SCONR_C_fix_u_M_a_Ea_params_dict['c_DOC'], 1e-8), LowerBound.apply(SCONR_C_fix_u_M_a_Ea_params_dict['c_MBC'], 1e-8)], 1))) #Create single diffusion matrix by diagonalizing constant noise scale parameters. 
    #diffusion_sqrt_single = torch.diag_embed(torch.sqrt(LowerBound.apply(torch.as_tensor([SCONR_C_fix_u_M_a_Ea_params_dict['c_SOC'], SCONR_C_fix_u_M_a_Ea_params_dict['c_DOC'], SCONR_C_fix_u_M_a_Ea_params_dict['c_MBC'], SCONR_C_fix_u_M_a_Ea_params_dict['c_CO2']]), 1e-8))) #Create single diffusion matrix by diagonalizing constant noise scale parameters.
    diffusion_sqrt = diffusion_sqrt_single.unsqueeze(1).expand(-1, T_SPAN_TENSOR.size(1), -1, -1) #Expand diffusion matrices across all paths and across discretized time steps.
    return drift, diffusion_sqrt

#SCONR-C fix a and Ea experiment
def drift_diffusion_SCONR_C_fix_a_Ea(C_PATH, T_SPAN_TENSOR, I_S_TENSOR, I_D_TENSOR, TEMP_TENSOR, TEMP_REF, SCONR_C_fix_a_Ea_params_dict):
    '''
    Returns SCONR fix a and Ea "constant diffusion parameterization" drift vectors and diffusion matrices.
    Expected SCONR_C_fix_a_Ea_params_dict = {'u_M': u_M, 'k_S_ref': k_S_ref, 'k_D_ref': k_D_ref, 'k_M_ref': k_M_ref, 'c_SOC': c_SOC, 'c_DOC': c_DOC, 'c_MBC': c_MBC}
    '''
    state_dim = 3 #SCON has three state variables in SOC, DOC, and MBC.
    SOC, DOC, MBC =  torch.chunk(C_PATH, state_dim, -1) #Partition SOC, DOC, and MBC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor. 
    drift = torch.empty_like(C_PATH, device = C_PATH.device) #Initiate tensor with same dims as C_PATH to assign drift.
    #Decay parameters are forced by temperature changes.
    k_S = arrhenius_temp_dep(SCONR_C_fix_a_Ea_params_dict['k_S_ref'], TEMP_TENSOR, 55, TEMP_REF) #Apply vectorized temperature-dependent transformation to k_S_ref.
    k_S = k_S.permute(2, 1, 0) #Get k_S into appropriate dimensions. 
    k_D = arrhenius_temp_dep(SCONR_C_fix_a_Ea_params_dict['k_D_ref'], TEMP_TENSOR, 48, TEMP_REF) #Apply vectorized temperature-dependent transformation to k_D_ref.
    k_D = k_D.permute(2, 1, 0) #Get k_D into appropriate dimensions.
    k_M = arrhenius_temp_dep(SCONR_C_fix_a_Ea_params_dict['k_M_ref'], TEMP_TENSOR, 48, TEMP_REF) #Apply vectorized temperature-dependent transformation to k_M_ref.
    k_M = k_M.permute(2, 1, 0) #Get k_M into appropriate dimensions.
    #Repeat and permute parameter values to match dimension sizes
    SCONR_C_fix_a_Ea_params_dict_rep = dict((k, v.repeat(1, T_SPAN_TENSOR.size(1), 1).permute(2, 1, 0)) for k, v in SCONR_C_fix_a_Ea_params_dict.items())    
    #Drift is calculated.
    drift_SOC = I_S_TENSOR + 0.5 * k_D * DOC + 0.5 * k_M * MBC - k_S * SOC
    drift_DOC = I_D_TENSOR + 0.5 * k_S * SOC + 0.5 * k_M * MBC - (SCONR_C_fix_a_Ea_params_dict_rep['u_M'] + k_D) * DOC
    drift_MBC = SCONR_C_fix_a_Ea_params_dict_rep['u_M'] * DOC - k_M * MBC
    #CO2 = (k_S * SOC * (1 - SCONR_C_fix_a_Ea_params_dict_rep['a_SD'])) + (k_D * DOC * (1 - SCONR_C_fix_a_Ea_params_dict_rep['a_DS'])) + (k_M * MBC * (1 - SCONR_C_fix_a_Ea_params_dict_rep['a_M'])) 
    #Assign elements to drift vector.
    drift[:, :, 0 : 1] = drift_SOC
    drift[:, :, 1 : 2] = drift_DOC
    drift[:, :, 2 : 3] = drift_MBC
    #drift[:, :, 3 : 4] = CO2 #CO2 is not a part of the drift. This is a hack for the explicit algebraic variable situation.
    #Diffusion matrix is assigned.
    diffusion_sqrt_single = torch.diag_embed(torch.sqrt(torch.stack([LowerBound.apply(SCONR_C_fix_a_Ea_params_dict['c_SOC'], 1e-8), LowerBound.apply(SCONR_C_fix_a_Ea_params_dict['c_DOC'], 1e-8), LowerBound.apply(SCONR_C_fix_a_Ea_params_dict['c_MBC'], 1e-8)], 1))) #Create single diffusion matrix by diagonalizing constant noise scale parameters. 
    #diffusion_sqrt_single = torch.diag_embed(torch.sqrt(LowerBound.apply(torch.as_tensor([SCONR_C_fix_a_Ea_params_dict['c_SOC'], SCONR_C_fix_a_Ea_params_dict['c_DOC'], SCONR_C_fix_a_Ea_params_dict['c_MBC'], SCONR_C_fix_a_Ea_params_dict['c_CO2']]), 1e-8))) #Create single diffusion matrix by diagonalizing constant noise scale parameters.
    diffusion_sqrt = diffusion_sqrt_single.unsqueeze(1).expand(-1, T_SPAN_TENSOR.size(1), -1, -1) #Expand diffusion matrices across all paths and across discretized time steps.
    return drift, diffusion_sqrt

#SCONR-C fix a experiment
def drift_diffusion_SCONR_C_fix_a(C_PATH, T_SPAN_TENSOR, I_S_TENSOR, I_D_TENSOR, TEMP_TENSOR, TEMP_REF, SCONR_C_fix_a_params_dict):
    '''
    Returns SCONR fix a "constant diffusion parameterization" drift vectors and diffusion matrices.
    Expected SCONR_C_fix_a_params_dict = {'u_M': u_M, 'k_S_ref': k_S_ref, 'k_D_ref': k_D_ref, 'k_M_ref': k_M_ref, 'Ea_S': Ea_S, 'Ea_D': Ea_D, 'Ea_M': Ea_M, 'c_SOC': c_SOC, 'c_DOC': c_DOC, 'c_MBC': c_MBC}
    '''
    state_dim = 3 #SCON has three state variables in SOC, DOC, and MBC.
    SOC, DOC, MBC =  torch.chunk(C_PATH, state_dim, -1) #Partition SOC, DOC, and MBC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor. 
    drift = torch.empty_like(C_PATH, device = C_PATH.device) #Initiate tensor with same dims as C_PATH to assign drift.
    #Decay parameters are forced by temperature changes.
    k_S = arrhenius_temp_dep(SCONR_C_fix_a_params_dict['k_S_ref'], TEMP_TENSOR, SCONR_C_fix_a_params_dict['Ea_S'], TEMP_REF) #Apply vectorized temperature-dependent transformation to k_S_ref.
    k_S = k_S.permute(2, 1, 0) #Get k_S into appropriate dimensions. 
    k_D = arrhenius_temp_dep(SCONR_C_fix_a_params_dict['k_D_ref'], TEMP_TENSOR, SCONR_C_fix_a_params_dict['Ea_D'], TEMP_REF) #Apply vectorized temperature-dependent transformation to k_D_ref.
    k_D = k_D.permute(2, 1, 0) #Get k_D into appropriate dimensions.
    k_M = arrhenius_temp_dep(SCONR_C_fix_a_params_dict['k_M_ref'], TEMP_TENSOR, SCONR_C_fix_a_params_dict['Ea_M'], TEMP_REF) #Apply vectorized temperature-dependent transformation to k_M_ref.
    k_M = k_M.permute(2, 1, 0) #Get k_M into appropriate dimensions.
    #Repeat and permute parameter values to match dimension sizes
    SCONR_C_fix_a_params_dict_rep = dict((k, v.repeat(1, T_SPAN_TENSOR.size(1), 1).permute(2, 1, 0)) for k, v in SCONR_C_fix_a_params_dict.items())    
    #Drift is calculated.
    drift_SOC = I_S_TENSOR + 0.5 * k_D * DOC + 0.5 * k_M * MBC - k_S * SOC
    drift_DOC = I_D_TENSOR + 0.5 * k_S * SOC + 0.5 * k_M * MBC - (SCONR_C_fix_a_params_dict_rep['u_M'] + k_D) * DOC
    drift_MBC = SCONR_C_fix_a_params_dict_rep['u_M'] * DOC - k_M * MBC
    #CO2 = (k_S * SOC * (1 - SCONR_C_fix_a_params_dict_rep['a_SD'])) + (k_D * DOC * (1 - SCONR_C_fix_a_params_dict_rep['a_DS'])) + (k_M * MBC * (1 - SCONR_C_fix_a_params_dict_rep['a_M'])) 
    #Assign elements to drift vector.
    drift[:, :, 0 : 1] = drift_SOC
    drift[:, :, 1 : 2] = drift_DOC
    drift[:, :, 2 : 3] = drift_MBC
    #drift[:, :, 3 : 4] = CO2 #CO2 is not a part of the drift. This is a hack for the explicit algebraic variable situation.
    #Diffusion matrix is assigned.
    diffusion_sqrt_single = torch.diag_embed(torch.sqrt(torch.stack([LowerBound.apply(SCONR_C_fix_a_params_dict['c_SOC'], 1e-8), LowerBound.apply(SCONR_C_fix_a_params_dict['c_DOC'], 1e-8), LowerBound.apply(SCONR_C_fix_a_params_dict['c_MBC'], 1e-8)], 1))) #Create single diffusion matrix by diagonalizing constant noise scale parameters. 
    #diffusion_sqrt_single = torch.diag_embed(torch.sqrt(LowerBound.apply(torch.as_tensor([SCONR_C_fix_a_params_dict['c_SOC'], SCONR_C_fix_a_params_dict['c_DOC'], SCONR_C_fix_a_params_dict['c_MBC'], SCONR_C_fix_a_params_dict['c_CO2']]), 1e-8))) #Create single diffusion matrix by diagonalizing constant noise scale parameters.
    diffusion_sqrt = diffusion_sqrt_single.unsqueeze(1).expand(-1, T_SPAN_TENSOR.size(1), -1, -1) #Expand diffusion matrices across all paths and across discretized time steps.
    return drift, diffusion_sqrt

#SCONR-C fix k experiment
def drift_diffusion_SCONR_C_fix_k(C_PATH, T_SPAN_TENSOR, I_S_TENSOR, I_D_TENSOR, TEMP_TENSOR, TEMP_REF, SCONR_C_fix_k_params_dict):
    '''
    Returns SCONR fix k "constant diffusion parameterization" drift vectors and diffusion matrices.
    Expected SCONR_C_fix_k_params_dict = {'u_M': u_M, 'a_SD': a_SD, 'a_DS': a_DS, 'a_MSC': a_MSC, 'Ea_S': Ea_S, 'Ea_D': Ea_D, 'Ea_M': Ea_M, 'c_SOC': c_SOC, 'c_DOC': c_DOC, 'c_MBC': c_MBC}
    '''
    state_dim = 3 #SCON has three state variables in SOC, DOC, and MBC.
    SOC, DOC, MBC =  torch.chunk(C_PATH, state_dim, -1) #Partition SOC, DOC, and MBC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor. 
    drift = torch.empty_like(C_PATH, device = C_PATH.device) #Initiate tensor with same dims as C_PATH to assign drift.
    #Decay parameters are forced by temperature changes.
    k_S = arrhenius_temp_dep(0.0005, TEMP_TENSOR, SCONR_C_fix_k_params_dict['Ea_S'], TEMP_REF) #Apply vectorized temperature-dependent transformation to k_S_ref.
    k_S = k_S.permute(2, 1, 0) #Get k_S into appropriate dimensions. 
    k_D = arrhenius_temp_dep(0.003, TEMP_TENSOR, SCONR_C_fix_k_params_dict['Ea_D'], TEMP_REF) #Apply vectorized temperature-dependent transformation to k_D_ref.
    k_D = k_D.permute(2, 1, 0) #Get k_D into appropriate dimensions.
    k_M = arrhenius_temp_dep(0.001, TEMP_TENSOR, SCONR_C_fix_k_params_dict['Ea_M'], TEMP_REF) #Apply vectorized temperature-dependent transformation to k_M_ref.
    k_M = k_M.permute(2, 1, 0) #Get k_M into appropriate dimensions.
    #Repeat and permute parameter values to match dimension sizes
    SCONR_C_fix_k_params_dict_rep = dict((k, v.repeat(1, T_SPAN_TENSOR.size(1), 1).permute(2, 1, 0)) for k, v in SCONR_C_fix_k_params_dict.items())    
    #Drift is calculated.
    drift_SOC = I_S_TENSOR + SCONR_C_fix_k_params_dict_rep['a_DS'] * k_D * DOC + SCONR_C_fix_k_params_dict_rep['a_MSC'] * k_M * MBC - k_S * SOC
    drift_DOC = I_D_TENSOR + SCONR_C_fix_k_params_dict_rep['a_SD'] * k_S * SOC + (1 - SCONR_C_fix_k_params_dict_rep['a_MSC']) * k_M * MBC - (SCONR_C_fix_k_params_dict_rep['u_M'] + k_D) * DOC
    drift_MBC = SCONR_C_fix_k_params_dict_rep['u_M'] * DOC - k_M * MBC
    #CO2 = (k_S * SOC * (1 - SCONR_C_fix_k_params_dict_rep['a_SD'])) + (k_D * DOC * (1 - SCONR_C_fix_k_params_dict_rep['a_DS'])) + (k_M * MBC * (1 - SCONR_C_fix_k_params_dict_rep['a_M'])) 
    #Assign elements to drift vector.
    drift[:, :, 0 : 1] = drift_SOC
    drift[:, :, 1 : 2] = drift_DOC
    drift[:, :, 2 : 3] = drift_MBC
    #drift[:, :, 3 : 4] = CO2 #CO2 is not a part of the drift. This is a hack for the explicit algebraic variable situation.
    #Diffusion matrix is assigned.
    diffusion_sqrt_single = torch.diag_embed(torch.sqrt(torch.stack([LowerBound.apply(SCONR_C_fix_k_params_dict['c_SOC'], 1e-8), LowerBound.apply(SCONR_C_fix_k_params_dict['c_DOC'], 1e-8), LowerBound.apply(SCONR_C_fix_k_params_dict['c_MBC'], 1e-8)], 1))) #Create single diffusion matrix by diagonalizing constant noise scale parameters. 
    #diffusion_sqrt_single = torch.diag_embed(torch.sqrt(LowerBound.apply(torch.as_tensor([SCONR_C_fix_k_params_dict['c_SOC'], SCONR_C_fix_k_params_dict['c_DOC'], SCONR_C_fix_k_params_dict['c_MBC'], SCONR_C_fix_k_params_dict['c_CO2']]), 1e-8))) #Create single diffusion matrix by diagonalizing constant noise scale parameters.
    diffusion_sqrt = diffusion_sqrt_single.unsqueeze(1).expand(-1, T_SPAN_TENSOR.size(1), -1, -1) #Expand diffusion matrices across all paths and across discretized time steps.
    return drift, diffusion_sqrt

#SCONR-C
def drift_diffusion_SCONR_C(C_PATH, T_SPAN_TENSOR, I_S_TENSOR, I_D_TENSOR, TEMP_TENSOR, TEMP_REF, SCONR_C_params_dict):
    '''
    Returns SCONR "constant diffusion parameterization" drift vectors and diffusion matrices.
    Expected SCONR_C_params_dict = {'u_M': u_M, 'a_SD': a_SD, 'a_DS': a_DS, 'a_MSC': a_MSC, 'k_S_ref': k_S_ref, 'k_D_ref': k_D_ref, 'k_M_ref': k_M_ref, 'Ea_S': Ea_S, 'Ea_D': Ea_D, 'Ea_M': Ea_M, 'c_SOC': c_SOC, 'c_DOC': c_DOC, 'c_MBC': c_MBC}
    '''
    state_dim = 3 #SCON has three state variables in SOC, DOC, and MBC.
    SOC, DOC, MBC =  torch.chunk(C_PATH, state_dim, -1) #Partition SOC, DOC, and MBC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor. 
    drift = torch.empty_like(C_PATH, device = C_PATH.device) #Initiate tensor with same dims as C_PATH to assign drift.
    #Decay parameters are forced by temperature changes.
    k_S = arrhenius_temp_dep(SCONR_C_params_dict['k_S_ref'], TEMP_TENSOR, SCONR_C_params_dict['Ea_S'], TEMP_REF) #Apply vectorized temperature-dependent transformation to k_S_ref.
    k_S = k_S.permute(2, 1, 0) #Get k_S into appropriate dimensions. 
    k_D = arrhenius_temp_dep(SCONR_C_params_dict['k_D_ref'], TEMP_TENSOR, SCONR_C_params_dict['Ea_D'], TEMP_REF) #Apply vectorized temperature-dependent transformation to k_D_ref.
    k_D = k_D.permute(2, 1, 0) #Get k_D into appropriate dimensions.
    k_M = arrhenius_temp_dep(SCONR_C_params_dict['k_M_ref'], TEMP_TENSOR, SCONR_C_params_dict['Ea_M'], TEMP_REF) #Apply vectorized temperature-dependent transformation to k_M_ref.
    k_M = k_M.permute(2, 1, 0) #Get k_M into appropriate dimensions.
    #Repeat and permute parameter values to match dimension sizes
    SCONR_C_params_dict_rep = dict((k, v.repeat(1, T_SPAN_TENSOR.size(1), 1).permute(2, 1, 0)) for k, v in SCONR_C_params_dict.items())    
    #Drift is calculated.
    drift_SOC = I_S_TENSOR + SCONR_C_params_dict_rep['a_DS'] * k_D * DOC + SCONR_C_params_dict_rep['a_MSC'] * k_M * MBC - k_S * SOC
    drift_DOC = I_D_TENSOR + SCONR_C_params_dict_rep['a_SD'] * k_S * SOC + (1 - SCONR_C_params_dict_rep['a_MSC']) * k_M * MBC - (SCONR_C_params_dict_rep['u_M'] + k_D) * DOC
    drift_MBC = SCONR_C_params_dict_rep['u_M'] * DOC - k_M * MBC
    #CO2 = (k_S * SOC * (1 - SCONR_C_params_dict_rep['a_SD'])) + (k_D * DOC * (1 - SCONR_C_params_dict_rep['a_DS'])) + (k_M * MBC * (1 - SCONR_C_params_dict_rep['a_M'])) 
    #Assign elements to drift vector.
    drift[:, :, 0 : 1] = drift_SOC
    drift[:, :, 1 : 2] = drift_DOC
    drift[:, :, 2 : 3] = drift_MBC
    #drift[:, :, 3 : 4] = CO2 #CO2 is not a part of the drift. This is a hack for the explicit algebraic variable situation.
    #Diffusion matrix is assigned.
    diffusion_sqrt_single = torch.diag_embed(torch.sqrt(torch.stack([LowerBound.apply(SCONR_C_params_dict['c_SOC'], 1e-8), LowerBound.apply(SCONR_C_params_dict['c_DOC'], 1e-8), LowerBound.apply(SCONR_C_params_dict['c_MBC'], 1e-8)], 1))) #Create single diffusion matrix by diagonalizing constant noise scale parameters. 
    #diffusion_sqrt_single = torch.diag_embed(torch.sqrt(LowerBound.apply(torch.as_tensor([SCONR_C_params_dict['c_SOC'], SCONR_C_params_dict['c_DOC'], SCONR_C_params_dict['c_MBC'], SCONR_C_params_dict['c_CO2']]), 1e-8))) #Create single diffusion matrix by diagonalizing constant noise scale parameters.
    diffusion_sqrt = diffusion_sqrt_single.unsqueeze(1).expand(-1, T_SPAN_TENSOR.size(1), -1, -1) #Expand diffusion matrices across all paths and across discretized time steps.
    return drift, diffusion_sqrt

#SCONR-SS
def drift_diffusion_SCONR_SS(C_PATH, T_SPAN_TENSOR, I_S_TENSOR, I_D_TENSOR, TEMP_TENSOR, TEMP_REF, SCONR_SS_params_dict):
    '''
    Returns SCONR "state scaling diffusion parameterization" drift vectors and diffusion matrices.
    Expected SCONR_SS_params_dict = {'u_M': u_M, 'a_SD': a_SD, 'a_DS': a_DS, 'a_MSC': a_MSC, 'k_S_ref': k_S_ref, 'k_D_ref': k_D_ref, 'k_M_ref': k_M_ref, 'Ea_S': Ea_S, 'Ea_D': Ea_D, 'Ea_M': Ea_M, 's_SOC': s_SOC, 's_DOC': s_DOC, 's_MBC': s_MBC}
    '''
    state_dim = 3 #SCON has three state variables in SOC, DOC, and MBC.
    SOC, DOC, MBC =  torch.chunk(C_PATH, state_dim, -1) #Partition SOC, DOC, MBC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor. 
    drift = torch.empty_like(C_PATH, device = C_PATH.device) #Initiate tensor with same dims as C_PATH to assign drift.
    diffusion_sqrt = torch.zeros([drift.size(0), drift.size(1), state_dim, state_dim], device = drift.device) #Create tensor to assign diffusion matrix elements.
    #diffusion_sqrt_diag = torch.empty_like(C_PATH, device = C_PATH.device) #Create tensor to assign diffusion matrix elements.
    #Decay parameters are forced by temperature changes.
    k_S = arrhenius_temp_dep(SCONR_SS_params_dict['k_S_ref'], TEMP_TENSOR, SCONR_SS_params_dict['Ea_S'], TEMP_REF) #Apply vectorized temperature-dependent transformation to k_S_ref.
    k_S = k_S.permute(2, 1, 0) #Get k_S into appropriate dimensions. 
    k_D = arrhenius_temp_dep(SCONR_SS_params_dict['k_D_ref'], TEMP_TENSOR, SCONR_SS_params_dict['Ea_D'], TEMP_REF) #Apply vectorized temperature-dependent transformation to k_D_ref.
    k_D = k_D.permute(2, 1, 0) #Get k_D into appropriate dimensions.
    k_M = arrhenius_temp_dep(SCONR_SS_params_dict['k_M_ref'], TEMP_TENSOR, SCONR_SS_params_dict['Ea_M'], TEMP_REF) #Apply vectorized temperature-dependent transformation to k_M_ref.
    k_M = k_M.permute(2, 1, 0) #Get k_M into appropriate dimensions.
    #Repeat and permute parameter values to match dimension sizes
    SCONR_SS_params_dict_rep = dict((k, v.repeat(1, T_SPAN_TENSOR.size(1), 1).permute(2, 1, 0)) for k, v in SCONR_SS_params_dict.items())    
    #Drift is calculated.
    drift_SOC = I_S_TENSOR + SCONR_SS_params_dict_rep['a_DS'] * k_D * DOC + SCONR_SS_params_dict_rep['a_MSC'] * k_M * MBC - k_S * SOC
    drift_DOC = I_D_TENSOR + SCONR_SS_params_dict_rep['a_SD'] * k_S * SOC + (1 - SCONR_SS_params_dict_rep['a_MSC']) * k_M * MBC - (SCONR_SS_params_dict_rep['u_M'] + k_D) * DOC
    drift_MBC = SCONR_SS_params_dict_rep['u_M'] * DOC - k_M * MBC
    #CO2 = (k_S * SOC * (1 - SCONR_SS_params_dict_rep['a_SD'])) + (k_D * DOC * (1 - SCONR_SS_params_dict_rep['a_DS'])) + (k_M * MBC * (1 - SCONR_SS_params_dict_rep['a_M'])) 
    #Assign elements to drift vector.
    drift[:, :, 0 : 1] = drift_SOC
    drift[:, :, 1 : 2] = drift_DOC
    drift[:, :, 2 : 3] = drift_MBC
    #drift[:, :, 3 : 4] = CO2 #CO2 is not a part of the drift. This is a hack for the explicit algebraic variable situation.
    #Diffusion matrix is assigned.
    diffusion_sqrt[:, :, 0 : 1, 0] = torch.sqrt(LowerBound.apply(SOC * SCONR_SS_params_dict_rep['s_SOC'], 1e-8)) #SOC diffusion standard deviation
    diffusion_sqrt[:, :, 1 : 2, 1] = torch.sqrt(LowerBound.apply(DOC * SCONR_SS_params_dict_rep['s_DOC'], 1e-8)) #DOC diffusion standard deviation
    diffusion_sqrt[:, :, 2 : 3, 2] = torch.sqrt(LowerBound.apply(MBC * SCONR_SS_params_dict_rep['s_MBC'], 1e-8)) #MBC diffusion standard deviation
    #diffusion_sqrt[:, :, 3 : 4, 3] = torch.sqrt(LowerBound.apply(CO2 * SCONR_SS_params_dict_rep['s_CO2'], 1e-8)) #CO2 diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 0 : 1] = torch.sqrt(LowerBound.apply(SOC * SCONR_SS_params_dict_rep['s_SOC'], 1e-8)) #SOC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 1 : 2] = torch.sqrt(LowerBound.apply(DOC * SCONR_SS_params_dict_rep['s_DOC'], 1e-8)) #DOC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 2 : 3] = torch.sqrt(LowerBound.apply(MBC * SCONR_SS_params_dict_rep['s_MBC'], 1e-8)) #MBC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 3 : 4] = torch.sqrt(LowerBound.apply(CO2 * SCONR_SS_params_dict_rep['s_CO2'], 1e-8)) #CO2 diffusion standard deviation
    #diffusion_sqrt = torch.diag_embed(diffusion_sqrt_diag)
    return drift, diffusion_sqrt

#SCON-C
def drift_diffusion_SCON_C(C_PATH, T_SPAN_TENSOR, I_S_TENSOR, I_D_TENSOR, TEMP_TENSOR, TEMP_REF, SCON_C_params_dict):
    '''
    Returns SCON "constant diffusion parameterization" drift vectors and diffusion matrices.
    Expected SCON_C_params_dict = {'u_M': u_M, 'a_SD': a_SD, 'a_DS': a_DS, 'a_M': a_M, 'a_MSC': a_MSC, 'k_S_ref': k_S_ref, 'k_D_ref': k_D_ref, 'k_M_ref': k_M_ref, 'Ea_S': Ea_S, 'Ea_D': Ea_D, 'Ea_M': Ea_M, 'c_SOC': c_SOC, 'c_DOC': c_DOC, 'c_MBC': c_MBC}
    '''
    state_dim = 3 #SCON has three state variables in SOC, DOC, and MBC.
    SOC, DOC, MBC =  torch.chunk(C_PATH, state_dim, -1) #Partition SOC, DOC, and MBC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor. 
    drift = torch.empty_like(C_PATH, device = C_PATH.device) #Initiate tensor with same dims as C_PATH to assign drift.
    #Decay parameters are forced by temperature changes.
    k_S = arrhenius_temp_dep(SCON_C_params_dict['k_S_ref'], TEMP_TENSOR, SCON_C_params_dict['Ea_S'], TEMP_REF) #Apply vectorized temperature-dependent transformation to k_S_ref.
    k_S = k_S.permute(2, 1, 0) #Get k_S into appropriate dimensions. 
    k_D = arrhenius_temp_dep(SCON_C_params_dict['k_D_ref'], TEMP_TENSOR, SCON_C_params_dict['Ea_D'], TEMP_REF) #Apply vectorized temperature-dependent transformation to k_D_ref.
    k_D = k_D.permute(2, 1, 0) #Get k_D into appropriate dimensions.
    k_M = arrhenius_temp_dep(SCON_C_params_dict['k_M_ref'], TEMP_TENSOR, SCON_C_params_dict['Ea_M'], TEMP_REF) #Apply vectorized temperature-dependent transformation to k_M_ref.
    k_M = k_M.permute(2, 1, 0) #Get k_M into appropriate dimensions.
    #Repeat and permute parameter values to match dimension sizes
    SCON_C_params_dict_rep = dict((k, v.repeat(1, T_SPAN_TENSOR.size(1), 1).permute(2, 1, 0)) for k, v in SCON_C_params_dict.items())    
    #Drift is calculated.
    drift_SOC = I_S_TENSOR + SCON_C_params_dict_rep['a_DS'] * k_D * DOC + SCON_C_params_dict_rep['a_M'] * SCON_C_params_dict_rep['a_MSC'] * k_M * MBC - k_S * SOC
    drift_DOC = I_D_TENSOR + SCON_C_params_dict_rep['a_SD'] * k_S * SOC + SCON_C_params_dict_rep['a_M'] * (1 - SCON_C_params_dict_rep['a_MSC']) * k_M * MBC - (SCON_C_params_dict_rep['u_M'] + k_D) * DOC
    drift_MBC = SCON_C_params_dict_rep['u_M'] * DOC - k_M * MBC
    #CO2 = (k_S * SOC * (1 - SCON_C_params_dict_rep['a_SD'])) + (k_D * DOC * (1 - SCON_C_params_dict_rep['a_DS'])) + (k_M * MBC * (1 - SCON_C_params_dict_rep['a_M'])) 
    #Assign elements to drift vector.
    drift[:, :, 0 : 1] = drift_SOC
    drift[:, :, 1 : 2] = drift_DOC
    drift[:, :, 2 : 3] = drift_MBC
    #drift[:, :, 3 : 4] = CO2 #CO2 is not a part of the drift. This is a hack for the explicit algebraic variable situation.
    #Diffusion matrix is assigned.
    diffusion_sqrt_single = torch.diag_embed(torch.sqrt(torch.stack([LowerBound.apply(SCON_C_params_dict['c_SOC'], 1e-8), LowerBound.apply(SCON_C_params_dict['c_DOC'], 1e-8), LowerBound.apply(SCON_C_params_dict['c_MBC'], 1e-8)], 1))) #Create single diffusion matrix by diagonalizing constant noise scale parameters. 
    #diffusion_sqrt_single = torch.diag_embed(torch.sqrt(LowerBound.apply(torch.as_tensor([SCON_C_params_dict['c_SOC'], SCON_C_params_dict['c_DOC'], SCON_C_params_dict['c_MBC'], SCON_C_params_dict['c_CO2']]), 1e-8))) #Create single diffusion matrix by diagonalizing constant noise scale parameters.
    diffusion_sqrt = diffusion_sqrt_single.unsqueeze(1).expand(-1, T_SPAN_TENSOR.size(1), -1, -1) #Expand diffusion matrices across all paths and across discretized time steps.
    return drift, diffusion_sqrt

#SCON-SS
def drift_diffusion_SCON_SS(C_PATH, T_SPAN_TENSOR, I_S_TENSOR, I_D_TENSOR, TEMP_TENSOR, TEMP_REF, SCON_SS_params_dict):
    '''
    Returns SCON "state scaling diffusion parameterization" drift vectors and diffusion matrices.
    Expected SCON_SS_params_dict = {'u_M': u_M, 'a_SD': a_SD, 'a_DS': a_DS, 'a_M': a_M, 'a_MSC': a_MSC, 'k_S_ref': k_S_ref, 'k_D_ref': k_D_ref, 'k_M_ref': k_M_ref, 'Ea_S': Ea_S, 'Ea_D': Ea_D, 'Ea_M': Ea_M, 's_SOC': s_SOC, 's_DOC': s_DOC, 's_MBC': s_MBC}
    '''
    state_dim = 3 #SCON has three state variables in SOC, DOC, and MBC.
    SOC, DOC, MBC =  torch.chunk(C_PATH, state_dim, -1) #Partition SOC, DOC, MBC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor. 
    drift = torch.empty_like(C_PATH, device = C_PATH.device) #Initiate tensor with same dims as C_PATH to assign drift.
    diffusion_sqrt = torch.zeros([drift.size(0), drift.size(1), state_dim, state_dim], device = drift.device) #Create tensor to assign diffusion matrix elements.
    #diffusion_sqrt_diag = torch.empty_like(C_PATH, device = C_PATH.device) #Create tensor to assign diffusion matrix elements.
    #Decay parameters are forced by temperature changes.
    k_S = arrhenius_temp_dep(SCON_SS_params_dict['k_S_ref'], TEMP_TENSOR, SCON_SS_params_dict['Ea_S'], TEMP_REF) #Apply vectorized temperature-dependent transformation to k_S_ref.
    k_S = k_S.permute(2, 1, 0) #Get k_S into appropriate dimensions. 
    k_D = arrhenius_temp_dep(SCON_SS_params_dict['k_D_ref'], TEMP_TENSOR, SCON_SS_params_dict['Ea_D'], TEMP_REF) #Apply vectorized temperature-dependent transformation to k_D_ref.
    k_D = k_D.permute(2, 1, 0) #Get k_D into appropriate dimensions.
    k_M = arrhenius_temp_dep(SCON_SS_params_dict['k_M_ref'], TEMP_TENSOR, SCON_SS_params_dict['Ea_M'], TEMP_REF) #Apply vectorized temperature-dependent transformation to k_M_ref.
    k_M = k_M.permute(2, 1, 0) #Get k_M into appropriate dimensions.
    #Repeat and permute parameter values to match dimension sizes
    SCON_SS_params_dict_rep = dict((k, v.repeat(1, T_SPAN_TENSOR.size(1), 1).permute(2, 1, 0)) for k, v in SCON_SS_params_dict.items())    
    #Drift is calculated.
    drift_SOC = I_S_TENSOR + SCON_SS_params_dict_rep['a_DS'] * k_D * DOC + SCON_SS_params_dict_rep['a_M'] * SCON_SS_params_dict_rep['a_MSC'] * k_M * MBC - k_S * SOC
    drift_DOC = I_D_TENSOR + SCON_SS_params_dict_rep['a_SD'] * k_S * SOC + SCON_SS_params_dict_rep['a_M'] * (1 - SCON_SS_params_dict_rep['a_MSC']) * k_M * MBC - (SCON_SS_params_dict_rep['u_M'] + k_D) * DOC
    drift_MBC = SCON_SS_params_dict_rep['u_M'] * DOC - k_M * MBC
    #CO2 = (k_S * SOC * (1 - SCON_SS_params_dict_rep['a_SD'])) + (k_D * DOC * (1 - SCON_SS_params_dict_rep['a_DS'])) + (k_M * MBC * (1 - SCON_SS_params_dict_rep['a_M'])) 
    #Assign elements to drift vector.
    drift[:, :, 0 : 1] = drift_SOC
    drift[:, :, 1 : 2] = drift_DOC
    drift[:, :, 2 : 3] = drift_MBC
    #drift[:, :, 3 : 4] = CO2 #CO2 is not a part of the drift. This is a hack for the explicit algebraic variable situation.
    #Diffusion matrix is assigned.
    diffusion_sqrt[:, :, 0 : 1, 0] = torch.sqrt(LowerBound.apply(SOC * SCON_SS_params_dict_rep['s_SOC'], 1e-8)) #SOC diffusion standard deviation
    diffusion_sqrt[:, :, 1 : 2, 1] = torch.sqrt(LowerBound.apply(DOC * SCON_SS_params_dict_rep['s_DOC'], 1e-8)) #DOC diffusion standard deviation
    diffusion_sqrt[:, :, 2 : 3, 2] = torch.sqrt(LowerBound.apply(MBC * SCON_SS_params_dict_rep['s_MBC'], 1e-8)) #MBC diffusion standard deviation
    #diffusion_sqrt[:, :, 3 : 4, 3] = torch.sqrt(LowerBound.apply(CO2 * SCON_SS_params_dict_rep['s_CO2'], 1e-8)) #CO2 diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 0 : 1] = torch.sqrt(LowerBound.apply(SOC * SCON_SS_params_dict_rep['s_SOC'], 1e-8)) #SOC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 1 : 2] = torch.sqrt(LowerBound.apply(DOC * SCON_SS_params_dict_rep['s_DOC'], 1e-8)) #DOC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 2 : 3] = torch.sqrt(LowerBound.apply(MBC * SCON_SS_params_dict_rep['s_MBC'], 1e-8)) #MBC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 3 : 4] = torch.sqrt(LowerBound.apply(CO2 * SCON_SS_params_dict_rep['s_CO2'], 1e-8)) #CO2 diffusion standard deviation
    #diffusion_sqrt = torch.diag_embed(diffusion_sqrt_diag)
    return drift, diffusion_sqrt

#SAWB-C
def drift_diffusion_SAWB_C(C_PATH, T_SPAN_TENSOR, I_S_TENSOR, I_D_TENSOR, TEMP_TENSOR, TEMP_REF, SAWB_C_params_dict):
    '''
    Returns SAWB "constant diffusion parameterization" drift vectors and diffusion matrices.
    Expected SAWB_C_params_dict = {'u_Q_ref': u_Q_ref, 'Q': Q, 'a_MSA': a_MSA, 'K_D': K_D, 'K_U': K_U, 'V_D_ref': V_D_ref, 'V_U_ref': V_U_ref, 'Ea_V_D': Ea_V_D, 'Ea_V_U': Ea_V_U, 'r_M': r_M, 'r_E': r_E, 'r_L': r_L, 'c_SOC': c_SOC, 'c_DOC': c_DOC, 'c_MBC': c_MBC, 'c_EEC': c_EEC}
    '''
    state_dim = 4 #SAWB and SAWB-ECA have four state variables in SOC, DOC, MBC, and EEC.
    SOC, DOC, MBC, EEC =  torch.chunk(C_PATH, state_dim, -1) #Partition SOC, DOC, MBC, EEC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor. 
    drift = torch.empty_like(C_PATH, device = C_PATH.device) #Initiate tensor with same dims as C_PATH to assign drift.
    #Decay parameters are forced by temperature changes.
    u_Q = linear_temp_dep(SAWB_C_params_dict['u_Q_ref'], TEMP_TENSOR, SAWB_C_params_dict['Q'], TEMP_REF) #Apply linear temperature-dependence to u_Q.
    u_Q.permute(2, 1, 0) #Get u_Q into appropriate dimensions. 
    V_D = arrhenius_temp_dep(SAWB_C_params_dict['V_D_ref'], TEMP_TENSOR, SAWB_C_params_dict['Ea_V_D'], TEMP_REF) #Apply vectorized temperature-dependent transformation to V_D.
    V_D.permute(2, 1, 0) #Get V_D into appropriate dimensions.
    V_U = arrhenius_temp_dep(SAWB_C_params_dict['V_U_ref'], TEMP_TENSOR, SAWB_C_params_dict['Ea_V_U'], TEMP_REF) #Apply vectorized temperature-dependent transformation to V_U.
    V_U.permute(2, 1, 0) #Get V_U into appropriate dimensions.
    #Repeat and permute parameter values to match dimension sizes
    SAWB_C_params_dict_rep = dict((k, v.repeat(1, T_SPAN_TENSOR.size(1), 1).permute(2, 1, 0)) for k, v in SAWB_C_params_dict.items()) 
    #Drift is calculated.
    drift_SOC = I_S_TENSOR + SAWB_C_params_dict_rep['a_MSA'] * SAWB_C_params_dict_rep['r_M'] * MBC - ((V_D * EEC * SOC) / (SAWB_C_params_dict_rep['K_D'] + SOC))
    drift_DOC = I_D_TENSOR + (1 - SAWB_C_params_dict_rep['a_MSA']) * SAWB_C_params_dict_rep['r_M'] * MBC + ((V_D * EEC * SOC) / (SAWB_C_params_dict_rep['K_D'] + SOC)) + SAWB_C_params_dict_rep['r_L'] * EEC - ((V_U * MBC * DOC) / (SAWB_C_params_dict_rep['K_U'] + DOC))
    drift_MBC = (u_Q * (V_U * MBC * DOC) / (SAWB_C_params_dict_rep['K_U'] + DOC)) - (SAWB_C_params_dict_rep['r_M'] + SAWB_C_params_dict_rep['r_E']) * MBC
    drift_EEC = SAWB_C_params_dict_rep['r_E'] * MBC - SAWB_C_params_dict_rep['r_L'] * EEC
    #CO2 = (1 - u_Q) * (V_U * MBC * DOC) / (SAWB_C_params_dict['K_U'] + DOC)
    #Assign elements to drift vector.
    drift[:, :, 0 : 1] = drift_SOC
    drift[:, :, 1 : 2] = drift_DOC
    drift[:, :, 2 : 3] = drift_MBC
    drift[:, :, 3 : 4] = drift_EEC
    #drift[:, :, 4 : 5] = CO2 #CO2 is not a part of the drift. This is a hack for the explicit algebraic variable situation.
    #Diffusion matrix is assigned.
    diffusion_sqrt_single = torch.diag_embed(torch.sqrt(torch.stack([LowerBound.apply(SAWB_C_params_dict['c_SOC'], 1e-8), LowerBound.apply(SAWB_C_params_dict['c_DOC'], 1e-8), LowerBound.apply(SAWB_C_params_dict['c_MBC'], 1e-8), LowerBound.apply(SAWB_C_params_dict['c_EEC'], 1e-8)]))) #Create single diffusion matrix by diagonalizing constant noise scale parameters.
    #diffusion_sqrt_single = torch.diag_embed(torch.sqrt(LowerBound.apply(torch.as_tensor([SAWB_C_params_dict['c_SOC'], SAWB_C_params_dict['c_DOC'], SAWB_C_params_dict['c_MBC'], SAWB_C_params_dict['c_EEC'], SAWB_C_params_dict['c_CO2']]), 1e-8))) #Create single diffusion matrix by diagonalizing constant noise scale parameters.    
    diffusion_sqrt = diffusion_sqrt_single.unsqueeze(1).expand(-1, T_SPAN_TENSOR.size(1), -1, -1) #Expand diffusion matrices across all paths and across discretized time steps.
    return drift, diffusion_sqrt

#SAWB-SS
def drift_diffusion_SAWB_SS(C_PATH, T_SPAN_TENSOR, I_S_TENSOR, I_D_TENSOR, TEMP_TENSOR, TEMP_REF, SAWB_SS_params_dict):
    '''
    Returns SAWB "state scaling diffusion parameterization" drift vectors and diffusion matrices.
    Expected SAWB_SS_params_dict = {'u_Q_ref': u_Q_ref, 'Q': Q, 'a_MSA': a_MSA, 'K_D': K_D, 'K_U': K_U, 'V_D_ref': V_D_ref, 'V_U_ref': V_U_ref, 'Ea_V_D': Ea_V_D, 'Ea_V_U': Ea_V_U, 'r_M': r_M, 'r_E': r_E, 'r_L': r_L, 's_SOC': s_SOC, 's_DOC': s_DOC, 's_MBC': s_MBC}
    '''
    state_dim = 4 #SAWB and SAWB-ECA have four state variables in SOC, DOC, MBC, and EEC.
    SOC, DOC, MBC, EEC =  torch.chunk(C_PATH, state_dim, -1) #Partition SOC, DOC, MBC, EEC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor. 
    drift = torch.empty_like(C_PATH, device = C_PATH.device) #Initiate tensor with same dims as C_PATH to assign drift.
    diffusion_sqrt = torch.zeros([drift.size(0), drift.size(1), state_dim, state_dim], device = drift.device) #Create tensor to assign diffusion matrix elements.
    #diffusion_sqrt_diag = torch.empty_like(C_PATH, device = C_PATH.device) #Create tensor to assign diffusion matrix elements.
    #Decay parameters are forced by temperature changes.
    u_Q = linear_temp_dep(SAWB_SS_params_dict['u_Q_ref'], TEMP_TENSOR, SAWB_SS_params_dict['Q'], TEMP_REF) #Apply linear temperature-dependence to u_Q.
    u_Q.permute(2, 1, 0) #Get u_Q into appropriate dimensions. 
    V_D = arrhenius_temp_dep(SAWB_SS_params_dict['V_D_ref'], TEMP_TENSOR, SAWB_SS_params_dict['Ea_V_D'], TEMP_REF) #Apply vectorized temperature-dependent transformation to V_D.
    V_D.permute(2, 1, 0) #Get V_D into appropriate dimensions.
    V_U = arrhenius_temp_dep(SAWB_SS_params_dict['V_U_ref'], TEMP_TENSOR, SAWB_SS_params_dict['Ea_V_U'], TEMP_REF) #Apply vectorized temperature-dependent transformation to V_U.
    V_U.permute(2, 1, 0) #Get V_U into appropriate dimensions.
    #Repeat and permute parameter values to match dimension sizes
    SAWB_SS_params_dict_rep_rep = dict((k, v.repeat(1, T_SPAN_TENSOR.size(1), 1).permute(2, 1, 0)) for k, v in SAWB_SS_params_dict_rep.items()) 
    #Drift is calculated.
    drift_SOC = I_S_TENSOR + SAWB_SS_params_dict_rep_rep['a_MSA'] * SAWB_SS_params_dict_rep_rep['r_M'] * MBC - ((V_D * EEC * SOC) / (SAWB_SS_params_dict_rep_rep['K_D'] + SOC))
    drift_DOC = I_D_TENSOR + (1 - SAWB_SS_params_dict_rep_rep['a_MSA']) * SAWB_SS_params_dict_rep_rep['r_M'] * MBC + ((V_D * EEC * SOC) / (SAWB_SS_params_dict_rep_rep['K_D'] + SOC)) + SAWB_SS_params_dict_rep_rep['r_L'] * EEC - ((V_U * MBC * DOC) / (SAWB_SS_params_dict_rep_rep['K_U'] + DOC))
    drift_MBC = (u_Q * (V_U * MBC * DOC) / (SAWB_SS_params_dict_rep_rep['K_U'] + DOC)) - (SAWB_SS_params_dict_rep_rep['r_M'] + SAWB_SS_params_dict_rep_rep['r_E']) * MBC
    drift_EEC = SAWB_SS_params_dict_rep_rep['r_E'] * MBC - SAWB_SS_params_dict_rep_rep['r_L'] * EEC
    #CO2 = (1 - u_Q) * (V_U * MBC * DOC) / (SAWB_SS_params_dict_rep['K_U'] + DOC)
    #Assign elements to drift vector.
    drift[:, :, 0 : 1] = drift_SOC
    drift[:, :, 1 : 2] = drift_DOC
    drift[:, :, 2 : 3] = drift_MBC
    drift[:, :, 3 : 4] = drift_EEC
    #drift[:, :, 4 : 5] = CO2 #CO2 is not a part of the drift. This is a hack for the explicit algebraic variable situation.
    #Diffusion matrix is assigned.
    diffusion_sqrt[:, :, 0 : 1, 0] = torch.sqrt(LowerBound.apply(SOC * SAWB_SS_params_dict_rep['s_SOC'], 1e-8)) #SOC diffusion standard deviation
    diffusion_sqrt[:, :, 1 : 2, 1] = torch.sqrt(LowerBound.apply(DOC * SAWB_SS_params_dict_rep['s_DOC'], 1e-8)) #DOC diffusion standard deviation
    diffusion_sqrt[:, :, 2 : 3, 2] = torch.sqrt(LowerBound.apply(MBC * SAWB_SS_params_dict_rep['s_MBC'], 1e-8)) #MBC diffusion standard deviation
    diffusion_sqrt[:, :, 3 : 4, 3] = torch.sqrt(LowerBound.apply(EEC * SAWB_SS_params_dict_rep['s_EEC'], 1e-8)) #EEC diffusion standard deviation
    #diffusion_sqrt[:, :, 4 : 5, 4] = torch.sqrt(LowerBound.apply(CO2 * SAWB_SS_params_dict_rep['s_CO2'], 1e-8)) #CO2 diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 0 : 1] = torch.sqrt(LowerBound.apply(SOC * SAWB_SS_params_dict_rep['s_SOC'], 1e-8)) #SOC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 1 : 2] = torch.sqrt(LowerBound.apply(DOC * SAWB_SS_params_dict_rep['s_DOC'], 1e-8)) #DOC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 2 : 3] = torch.sqrt(LowerBound.apply(MBC * SAWB_SS_params_dict_rep['s_MBC'], 1e-8)) #MBC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 3 : 4] = torch.sqrt(LowerBound.apply(EEC * SAWB_SS_params_dict_rep['s_EEC'], 1e-8)) #EEC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 4 : 5] = torch.sqrt(LowerBound.apply(CO2 * SAWB_SS_params_dict_rep['s_CO2'], 1e-8)) #CO2 diffusion standard deviation
    #diffusion_sqrt = torch.diag_embed(diffusion_sqrt_diag)
    return drift, diffusion_sqrt

#SAWB-ECA-C
def drift_diffusion_SAWB_ECA_C(C_PATH, T_SPAN_TENSOR, I_S_TENSOR, I_D_TENSOR, TEMP_TENSOR, TEMP_REF, SAWB_ECA_C_params_dict):
    '''
    Returns SAWB-ECA "constant diffusion parameterization" drift vectors and diffusion matrices.
    Expected SAWB_ECA_C_params_dict = {'u_Q_ref': u_Q_ref, 'Q': Q, 'a_MSA': a_MSA, 'K_DE': K_DE, 'K_UE': K_UE, 'V_DE_ref': V_DE_ref, 'V_UE_ref': V_UE_ref, 'Ea_V_DE': Ea_V_DE, 'Ea_V_UE': Ea_V_UE, 'r_M': r_M, 'r_E': r_E, 'r_L': r_L, 'c_SOC': c_SOC, 'c_DOC': c_DOC, 'c_MBC': c_MBC, 'c_EEC': c_EEC}
    '''
    state_dim = 4 #SAWB and SAWB-ECA have four state variables in SOC, DOC, MBC, and EEC.
    SOC, DOC, MBC, EEC =  torch.chunk(C_PATH, state_dim, -1) #Partition SOC, DOC, MBC, EEC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor.
    drift = torch.empty_like(C_PATH, device = C_PATH.device) #Initiate tensor with same dims as C_PATH to assign drift.
    #Decay parameters are forced by temperature changes.
    u_Q = linear_temp_dep(SAWB_ECA_C_params_dict['u_Q_ref'], TEMP_TENSOR, SAWB_ECA_C_params_dict['Q'], TEMP_REF) #Apply linear temperature-dependence to u_Q.
    u_Q.permute(2, 1, 0) #Get u_Q into appropriate dimensions.    
    V_DE = arrhenius_temp_dep(SAWB_ECA_C_params_dict['V_DE_ref'], TEMP_TENSOR, SAWB_ECA_C_params_dict['Ea_V_DE'], TEMP_REF) #Apply vectorized temperature-dependent transformation to V_DE.
    V_DE.permute(2, 1, 0) #Get V_DE into appropriate dimensions.     
    V_UE = arrhenius_temp_dep(SAWB_ECA_C_params_dict['V_UE_ref'], TEMP_TENSOR, SAWB_ECA_C_params_dict['Ea_V_UE'], TEMP_REF) #Apply vectorized temperature-dependent transformation to V_UE.
    V_UE.permute(2, 1, 0) #Get V_UE into appropriate dimensions.
    #Repeat and permute parameter values to match dimension sizes
    SAWB_ECA_C_params_dict_rep = dict((k, v.repeat(1, T_SPAN_TENSOR.size(1), 1).permute(2, 1, 0)) for k, v in SAWB_ECA_C_params_dict_rep.items())
    #Drift is calculated.
    drift_SOC = I_S_TENSOR + SAWB_ECA_C_params_dict_rep['a_MSA'] * SAWB_ECA_C_params_dict_rep['r_M'] * MBC - ((V_DE * EEC * SOC) / (SAWB_ECA_C_params_dict_rep['K_DE'] + EEC + SOC))
    drift_DOC = I_D_TENSOR + (1 - SAWB_ECA_C_params_dict_rep['a_MSA']) * SAWB_ECA_C_params_dict_rep['r_M'] * MBC + ((V_DE * EEC * SOC) / (SAWB_ECA_C_params_dict_rep['K_DE'] + EEC + SOC)) + SAWB_ECA_C_params_dict_rep['r_L'] * EEC - ((V_UE * MBC * DOC) / (SAWB_ECA_C_params_dict_rep['K_UE'] + MBC + DOC))
    drift_MBC = (u_Q * (V_UE * MBC * DOC) / (SAWB_ECA_C_params_dict_rep['K_UE'] + MBC + DOC)) - (SAWB_ECA_C_params_dict_rep['r_M'] + SAWB_ECA_C_params_dict_rep['r_E']) * MBC
    drift_EEC = SAWB_ECA_C_params_dict_rep['r_E'] * MBC - SAWB_ECA_C_params_dict_rep['r_L'] * EEC
    #CO2 = (1 - u_Q) * (V_UE * MBC * DOC) / (SAWB_ECA_C_params_dict_rep['K_UE'] + MBC + DOC)
    #Assign elements to drift vector.
    drift[:, :, 0 : 1] = drift_SOC
    drift[:, :, 1 : 2] = drift_DOC
    drift[:, :, 2 : 3] = drift_MBC
    drift[:, :, 3 : 4] = drift_EEC
    #drift[:, :, 4 : 5] = CO2 #CO2 is not a part of the drift. This is a hack for the explicit algebraic variable situation.
    #Diffusion matrix is assigned.
    diffusion_sqrt_single = torch.diag_embed(torch.sqrt(torch.stack([LowerBound.apply(SAWB_ECA_C_params_dict['c_SOC'], 1e-8), LowerBound.apply(SAWB_ECA_C_params_dict['c_DOC'], 1e-8), LowerBound.apply(SAWB_ECA_C_params_dict['c_MBC'], 1e-8), LowerBound.apply(SAWB_ECA_C_params_dict['c_EEC'], 1e-8)]))) #Create single diffusion matrix by diagonalizing constant noise scale parameters.
    #diffusion_sqrt_single = torch.diag_embed(torch.sqrt(LowerBound.apply(torch.as_tensor([SAWB_ECA_C_params_dict['c_SOC'], SAWB_ECA_C_params_dict['c_DOC'], SAWB_ECA_C_params_dict['c_MBC'], SAWB_ECA_C_params_dict['c_EEC'], SAWB_ECA_C_params_dict['c_CO2']]), 1e-8))) #Create single diffusion matrix by diagonalizing constant noise scale parameters.
    diffusion_sqrt = diffusion_sqrt_single.expand(drift.size(0), drift.size(1), state_dim, state_dim) #Expand diffusion matrices across all paths and across discretized time steps.
    return drift, diffusion_sqrt

#SAWB-ECA-SS
def drift_diffusion_SAWB_ECA_SS(C_PATH, T_SPAN_TENSOR, I_S_TENSOR, I_D_TENSOR, TEMP_TENSOR, TEMP_REF, SAWB_ECA_SS_params_dict):
    '''
    Returns SAWB-ECA "state scaling diffusion parameterization" drift vectors and diffusion matrices.
    Expected SAWB_ECA_SS_params_dict = {'u_Q_ref': u_Q_ref, 'Q': Q, 'a_MSA': a_MSA, 'K_DE': K_DE, 'K_UE': K_UE, 'V_DE_ref': V_DE_ref, 'V_UE_ref': V_UE_ref, 'Ea_V_DE': Ea_V_DE, 'Ea_V_UE': Ea_V_UE, 'r_M': r_M, 'r_E': r_E, 'r_L': r_L, 's_SOC': s_SOC, 's_DOC': s_DOC, 's_MBC': s_MBC, 's_EEC': s_EEC}
    '''
    state_dim = 4 #SAWB and SAWB-ECA have four state variables in SOC, DOC, MBC, and EEC.
    SOC, DOC, MBC, EEC =  torch.chunk(C_PATH, state_dim, -1) #Partition SOC, DOC, MBC, EEC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor. 
    drift = torch.empty_like(C_PATH, device = C_PATH.device) #Initiate tensor with same dims as C_PATH to assign drift.
    diffusion_sqrt = torch.zeros([drift.size(0), drift.size(1), state_dim, state_dim], device = drift.device) #Create tensor to assign diffusion matrix elements.
    #diffusion_sqrt_diag = torch.empty_like(C_PATH, device = C_PATH.device) #Create tensor to assign diffusion matrix elements.
    #Decay parameters are forced by temperature changes.
    u_Q = linear_temp_dep(SAWB_ECA_SS_params_dict['u_Q_ref'], TEMP_TENSOR, SAWB_ECA_SS_params_dict['Q'], TEMP_REF) #Apply linear temperature-dependence to u_Q.
    u_Q.permute(2, 1, 0) #Get u_Q into appropriate dimensions.    
    V_DE = arrhenius_temp_dep(SAWB_ECA_SS_params_dict['V_DE_ref'], TEMP_TENSOR, SAWB_ECA_SS_params_dict['Ea_V_DE'], TEMP_REF) #Apply vectorized temperature-dependent transformation to V_DE.
    V_DE.permute(2, 1, 0) #Get V_DE into appropriate dimensions.     
    V_UE = arrhenius_temp_dep(SAWB_ECA_SS_params_dict['V_UE_ref'], TEMP_TENSOR, SAWB_ECA_SS_params_dict['Ea_V_UE'], TEMP_REF) #Apply vectorized temperature-dependent transformation to V_UE.
    V_UE.permute(2, 1, 0) #Get V_UE into appropriate dimensions.
    #Repeat and permute parameter values to match dimension sizes
    SAWB_ECA_SS_params_dict_rep = dict((k, v.repeat(1, T_SPAN_TENSOR.size(1), 1).permute(2, 1, 0)) for k, v in SAWB_ECA_SS_params_dict_rep.items())
    #Drift is calculated.
    drift_SOC = I_S_TENSOR + SAWB_ECA_SS_params_dict_rep['a_MSA'] * SAWB_ECA_SS_params_dict_rep['r_M'] * MBC - ((V_DE * EEC * SOC) / (SAWB_ECA_SS_params_dict_rep['K_DE'] + EEC + SOC))
    drift_DOC = I_D_TENSOR + (1 - SAWB_ECA_SS_params_dict_rep['a_MSA']) * SAWB_ECA_SS_params_dict_rep['r_M'] * MBC + ((V_DE * EEC * SOC) / (SAWB_ECA_SS_params_dict_rep['K_DE'] + EEC + SOC)) + SAWB_ECA_SS_params_dict_rep['r_L'] * EEC - ((V_UE * MBC * DOC) / (SAWB_ECA_SS_params_dict_rep['K_UE'] + MBC + DOC))
    drift_MBC = (u_Q * (V_UE * MBC * DOC) / (SAWB_ECA_SS_params_dict_rep['K_UE'] + MBC + DOC)) - (SAWB_ECA_SS_params_dict_rep['r_M'] + SAWB_ECA_SS_params_dict_rep['r_E']) * MBC
    drift_EEC = SAWB_ECA_SS_params_dict_rep['r_E'] * MBC - SAWB_ECA_SS_params_dict_rep['r_L'] * EEC
    #CO2 = (1 - u_Q) * (V_UE * MBC * DOC) / (SAWB_ECA_SS_params_dict_rep['K_UE'] + MBC + DOC)
    #Assign elements to drift vector.
    drift[:, :, 0 : 1] = drift_SOC
    drift[:, :, 1 : 2] = drift_DOC
    drift[:, :, 2 : 3] = drift_MBC
    drift[:, :, 3 : 4] = drift_EEC
    #drift[:, :, 4 : 5] = CO2 #CO2 is not a part of the drift. This is a hack for the explicit algebraic variable situation.
    #Diffusion matrix is assigned.
    diffusion_sqrt[:, :, 0 : 1, 0] = torch.sqrt(LowerBound.apply(SOC * SAWB_ECA_SS_params_dict_rep['s_SOC'], 1e-8)) #SOC diffusion standard deviation
    diffusion_sqrt[:, :, 1 : 2, 1] = torch.sqrt(LowerBound.apply(DOC * SAWB_ECA_SS_params_dict_rep['s_DOC'], 1e-8)) #DOC diffusion standard deviation
    diffusion_sqrt[:, :, 2 : 3, 2] = torch.sqrt(LowerBound.apply(MBC * SAWB_ECA_SS_params_dict_rep['s_MBC'], 1e-8)) #MBC diffusion standard deviation
    diffusion_sqrt[:, :, 3 : 4, 3] = torch.sqrt(LowerBound.apply(EEC * SAWB_ECA_SS_params_dict_rep['s_EEC'], 1e-8)) #EEC diffusion standard deviation
    #diffusion_sqrt[:, :, 4 : 5, 4] = torch.sqrt(LowerBound.apply(CO2 * SAWB_ECA_SS_params_dict_rep['s_CO2'], 1e-8)) #CO2 diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 0 : 1] = torch.sqrt(LowerBound.apply(SOC * SAWB_ECA_SS_params_dict_rep['s_SOC'], 1e-8)) #SOC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 1 : 2] = torch.sqrt(LowerBound.apply(DOC * SAWB_ECA_SS_params_dict_rep['s_DOC'], 1e-8)) #DOC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 2 : 3] = torch.sqrt(LowerBound.apply(MBC * SAWB_ECA_SS_params_dict_rep['s_MBC'], 1e-8)) #MBC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 3 : 4] = torch.sqrt(LowerBound.apply(EEC * SAWB_ECA_SS_params_dict_rep['s_EEC'], 1e-8)) #EEC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 4 : 5] = torch.sqrt(LowerBound.apply(CO2 * SAWB_ECA_SS_params_dict_rep['s_CO2'], 1e-8)) #CO2 diffusion standard deviation
    #diffusion_sqrt = torch.diag_embed(diffusion_sqrt_diag)
    return drift, diffusion_sqrt
