import torch
import numpy as np
from obs_and_flow_classes_and_functions import LowerBound

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

def temp_gen(t, temp_ref):
    temp = temp_ref + t / (20 * 24 * 365) + 10 * torch.sin((2 * np.pi / 24) * t) + 10 * torch.sin((2 * np.pi / (24 * 365)) * t)
    return temp

def arrhenius_temp_dep(parameter, temp, Ea, temp_ref):
    '''
    For a parameter with Arrhenius temperature dependence, returns the transformed parameter value.
    0.008314 is the gas constant. Temperatures are in K.
    '''
    decayed_parameter = parameter * torch.exp(-Ea / 0.008314 * (1 / temp - 1 / temp_ref))
    return decayed_parameter

def linear_temp_dep(parameter, temp, Q, temp_ref):
    '''
    For a parameter with linear temperature dependence, returns the transformed parameter value.
    Q is the slope of the temperature dependence and is a varying parameter.
    Temperatures are in K.
    '''
    modified_parameter = parameter - Q * (temp - temp_ref)
    return modified_parameter

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
    print(D_0)
    S_0 = (SOC_input + D_0 * (SCON_params_dict['a_DS'] * SCON_params_dict['k_D_ref'] + SCON_params_dict['u_M'] * SCON_params_dict['a_M'] * SCON_params_dict['a_MSC'])) / SCON_params_dict['k_S_ref']
    print(S_0)    
    M_0 = SCON_params_dict['u_M'] * D_0 / SCON_params_dict['k_M_ref']
    print(M_0)
    C_0_vector = torch.as_tensor([S_0, D_0, M_0])    
    #CO2_0 = SCON_params_dict['k_S_ref'] * S_0 * (1 - SCON_params_dict['a_SD']) + SCON_params_dict['k_D_ref'] * D_0 * (1 - SCON_params_dict['a_DS']) + SCON_params_dict['k_M_ref'] * M_0 * (1 - SCON_params_dict['a_M'])
    #C_0_vector = torch.as_tensor([S_0, D_0, M_0, CO2_0])
    return C_0_vector

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
    C_0_vector = torch.as_tensor([S_0, D_0, M_0, E_0])    
    #E_0 = -((SAWB_params_dict['r_E'] * SAWB_params_dict['u_Q_ref'] * (SOC_input + DOC_input)) / (SAWB_params_dict['r_L'] * (SAWB_params_dict['r_E'] + SAWB_params_dict['r_M']) * (SAWB_params_dict['u_Q_ref'] - 1)))
    #CO2_0 = (1 - SAWB_params_dict['u_Q_ref']) * (SAWB_params_dict['V_U_ref'] * M_0 * D_0) / (SAWB_params_dict['K_U'] + D_0)
    #C_0_vector = torch.as_tensor([S_0, D_0, M_0, E_0, CO2_0])
    return C_0_vector

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
    C_0_vector = torch.as_tensor([S_0, D_0, M_0, E_0])    
    #E_0 = -((SAWB_params_dict['r_E'] * SAWB_params_dict['u_Q_ref'] * (SOC_input + DOC_input)) / (SAWB_params_dict['r_L'] * (SAWB_params_dict['r_E'] + SAWB_params_dict['r_M']) * (SAWB_params_dict['u_Q_ref'] - 1)))
    #CO2_0 = (1 - SAWB_ECA_params_dict['u_Q_ref']) * SAWB_ECA_params_dict['V_UE_ref'] * M_0 * D_0 / (SAWB_ECA_params_dict['K_UE'] + M_0 + D_0)
    #C_0_vector = torch.as_tensor([S_0, D_0, M_0, E_0, CO2_0])
    return C_0_vector

####################################################
##STOCHASTIC DIFFERENTIAL EQUATION MODEL FUNCTIONS##
#################################################### 

#SCON-C
def drift_diffusion_SCON_C(C_PATH, T_SPAN_TENSOR, I_S_TENSOR, I_D_TENSOR, SCON_C_params_dict, temp_gen, temp_ref):
    '''
    Returns SCON "constant diffusion parameterization" drift vectors and diffusion matrices.
    current_temp is output from temp_gen function. 
    Expected SCON_C_params_dict = {'u_M': u_M, 'a_SD': a_SD, 'a_DS': a_DS, 'a_M': a_M, 'a_MSC': a_MSC, 'k_S_ref': k_S_ref, 'k_D_ref': k_D_ref, 'k_M_ref': k_M_ref, 'Ea_S': Ea_S, 'Ea_D': Ea_D, 'Ea_M': Ea_M, 'c_SOC': c_SOC, 'c_DOC': c_DOC, 'c_MBC': c_MBC}
    '''
    state_dim = 3 #SCON has three state variables in SOC, DOC, and MBC.
    SOC, DOC, MBC =  torch.chunk(C_PATH, state_dim, -1) #Partition SOC, DOC, and MBC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor. 
    current_temp = temp_gen(T_SPAN_TENSOR, temp_ref) #Obtain temperature function vector across span of times.
    drift = torch.empty_like(C_PATH, device = C_PATH.device) #Initiate tensor with same dims as C_PATH to assign drift.
    #Decay parameters are forced by temperature changes.
    k_S = arrhenius_temp_dep(SCON_C_params_dict['k_S_ref'], current_temp, SCON_C_params_dict['Ea_S'], temp_ref) #Apply vectorized temperature-dependent transformation to k_S_ref.
    k_D = arrhenius_temp_dep(SCON_C_params_dict['k_D_ref'], current_temp, SCON_C_params_dict['Ea_D'], temp_ref) #Apply vectorized temperature-dependent transformation to k_D_ref.
    k_M = arrhenius_temp_dep(SCON_C_params_dict['k_M_ref'], current_temp, SCON_C_params_dict['Ea_M'], temp_ref) #Apply vectorized temperature-dependent transformation to k_M_ref.
    #Drift is calculated.
    drift_SOC = I_S_TENSOR + SCON_C_params_dict['a_DS'] * k_D * DOC + SCON_C_params_dict['a_M'] * SCON_C_params_dict['a_MSC'] * k_M * MBC - k_S * SOC
    drift_DOC = I_D_TENSOR + SCON_C_params_dict['a_SD'] * k_S * SOC + SCON_C_params_dict['a_M'] * (1 - SCON_C_params_dict['a_MSC']) * k_M * MBC - (SCON_C_params_dict['u_M'] + k_D) * DOC
    drift_MBC = SCON_C_params_dict['u_M'] * DOC - k_M * MBC
    #CO2 = (k_S * SOC * (1 - SCON_C_params_dict['a_SD'])) + (k_D * DOC * (1 - SCON_C_params_dict['a_DS'])) + (k_M * MBC * (1 - SCON_C_params_dict['a_M'])) 
    #Assign elements to drift vector.
    drift[:, :, 0 : 1] = drift_SOC
    drift[:, :, 1 : 2] = drift_DOC
    drift[:, :, 2 : 3] = drift_MBC
    #drift[:, :, 3 : 4] = CO2 #CO2 is not a part of the drift. This is a hack for the explicit algebraic variable situation.
    #Diffusion matrix is assigned.
    diffusion_sqrt_single = torch.diag(torch.sqrt(LowerBound.apply(torch.as_tensor([SCON_C_params_dict['c_SOC'], SCON_C_params_dict['c_DOC'], SCON_C_params_dict['c_MBC']]), 1e-6))) #Create single diffusion matrix by diagonalizing constant noise scale parameters.    
    #diffusion_sqrt_single = torch.diag(torch.sqrt(LowerBound.apply(torch.as_tensor([SCON_C_params_dict['c_SOC'], SCON_C_params_dict['c_DOC'], SCON_C_params_dict['c_MBC'], SCON_C_params_dict['c_CO2']]), 1e-6))) #Create single diffusion matrix by diagonalizing constant noise scale parameters.
    diffusion_sqrt = diffusion_sqrt_single.expand(drift.size(0), drift.size(1), state_dim, state_dim) #Expand diffusion matrices across all paths and across discretized time steps.
    return drift, diffusion_sqrt

#SCON-SS
def drift_diffusion_SCON_SS(C_PATH, T_SPAN_TENSOR, I_S_TENSOR, I_D_TENSOR, SCON_SS_params_dict, temp_gen, temp_ref):
    '''
    Returns SCON "state scaling diffusion parameterization" drift vectors and diffusion matrices.
    current_temp is output from temp_gen function. 
    Expected SCON_SS_params_dict = {'u_M': u_M, 'a_SD': a_SD, 'a_DS': a_DS, 'a_M': a_M, 'a_MSC': a_MSC, 'k_S_ref': k_S_ref, 'k_D_ref': k_D_ref, 'k_M_ref': k_M_ref, 'Ea_S': Ea_S, 'Ea_D': Ea_D, 'Ea_M': Ea_M, 's_SOC': s_SOC, 's_DOC': s_DOC, 's_MBC': s_MBC}
    '''
    state_dim = 3 #SCON has three state variables in SOC, DOC, and MBC.
    SOC, DOC, MBC =  torch.chunk(C_PATH, state_dim, -1) #Partition SOC, DOC, MBC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor. 
    current_temp = temp_gen(T_SPAN_TENSOR, temp_ref) #Obtain temperature function vector across span of times.
    drift = torch.empty_like(C_PATH, device = C_PATH.device) #Initiate tensor with same dims as C_PATH to assign drift.
    diffusion_sqrt = torch.zeros([drift.size(0), drift.size(1), state_dim, state_dim], device = drift.device) #Create tensor to assign diffusion matrix elements. Diffusion exists for explicit algebraic variable CO2.
    #diffusion_sqrt_diag = torch.empty_like(C_PATH, device = C_PATH.device) #Create tensor to assign diffusion matrix elements.
    #Decay parameters are forced by temperature changes.
    k_S = arrhenius_temp_dep(SCON_SS_params_dict['k_S_ref'], current_temp, SCON_SS_params_dict['Ea_S'], temp_ref) #Apply vectorized temperature-dependent transformation to k_S_ref.
    k_D = arrhenius_temp_dep(SCON_SS_params_dict['k_D_ref'], current_temp, SCON_SS_params_dict['Ea_D'], temp_ref) #Apply vectorized temperature-dependent transformation to k_D_ref.
    k_M = arrhenius_temp_dep(SCON_SS_params_dict['k_M_ref'], current_temp, SCON_SS_params_dict['Ea_M'], temp_ref) #Apply vectorized temperature-dependent transformation to k_M_ref.
    #Drift is calculated.
    drift_SOC = I_S_TENSOR + SCON_SS_params_dict['a_DS'] * k_D * DOC + SCON_SS_params_dict['a_M'] * SCON_SS_params_dict['a_MSC'] * k_M * MBC - k_S * SOC
    drift_DOC = I_D_TENSOR + SCON_SS_params_dict['a_SD'] * k_S * SOC + SCON_SS_params_dict['a_M'] * (1 - SCON_SS_params_dict['a_MSC']) * k_M * MBC - (SCON_SS_params_dict['u_M'] + k_D) * DOC
    drift_MBC = SCON_SS_params_dict['u_M'] * DOC - k_M * MBC
    #CO2 = (k_S * SOC * (1 - SCON_SS_params_dict['a_SD'])) + (k_D * DOC * (1 - SCON_SS_params_dict['a_DS'])) + (k_M * MBC * (1 - SCON_SS_params_dict['a_M'])) 
    #Assign elements to drift vector.
    drift[:, :, 0 : 1] = drift_SOC
    drift[:, :, 1 : 2] = drift_DOC
    drift[:, :, 2 : 3] = drift_MBC
    #drift[:, :, 3 : 4] = CO2 #CO2 is not a part of the drift. This is a hack for the explicit algebraic variable situation.
    #Diffusion matrix is assigned.
    diffusion_sqrt[:, :, 0 : 1, 0] = torch.sqrt(LowerBound.apply(SOC * SCON_SS_params_dict['s_SOC'], 1e-6)) #SOC diffusion standard deviation
    diffusion_sqrt[:, :, 1 : 2, 1] = torch.sqrt(LowerBound.apply(DOC * SCON_SS_params_dict['s_DOC'], 1e-6)) #DOC diffusion standard deviation
    diffusion_sqrt[:, :, 2 : 3, 2] = torch.sqrt(LowerBound.apply(MBC * SCON_SS_params_dict['s_MBC'], 1e-6)) #MBC diffusion standard deviation
    #diffusion_sqrt[:, :, 3 : 4, 3] = torch.sqrt(LowerBound.apply(CO2 * SCON_SS_params_dict['s_CO2'], 1e-6)) #CO2 diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 0 : 1] = torch.sqrt(LowerBound.apply(SOC * SCON_SS_params_dict['s_SOC'], 1e-6)) #SOC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 1 : 2] = torch.sqrt(LowerBound.apply(DOC * SCON_SS_params_dict['s_DOC'], 1e-6)) #DOC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 2 : 3] = torch.sqrt(LowerBound.apply(MBC * SCON_SS_params_dict['s_MBC'], 1e-6)) #MBC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 3 : 4] = torch.sqrt(LowerBound.apply(CO2 * SCON_SS_params_dict['s_CO2'], 1e-6)) #CO2 diffusion standard deviation
    #diffusion_sqrt = torch.diag_embed(diffusion_sqrt_diag)
    return drift, diffusion_sqrt

#SAWB-c
def drift_diffusion_SAWB_C(C_PATH, T_SPAN_TENSOR, I_S_TENSOR, I_D_TENSOR, SAWB_C_params_dict, temp_gen, temp_ref):
    '''
    Returns SAWB "constant diffusion parameterization" drift vectors and diffusion matrices.
    current_temp is output from temp_gen function. 
    Expected SAWB_C_params_dict = {'u_Q_ref': u_Q_ref, 'Q': Q, 'a_MSA': a_MSA, 'K_D': K_D, 'K_U': K_U, 'V_D_ref': V_D_ref, 'V_U_ref': V_U_ref, 'Ea_V_D': Ea_V_D, 'Ea_V_U': Ea_V_U, 'r_M': r_M, 'r_E': r_E, 'r_L': r_L, 'c_SOC': c_SOC, 'c_DOC': c_DOC, 'c_MBC': c_MBC, 'c_EEC': c_EEC}
    '''
    state_dim = 4 #SAWB and SAWB-ECA have four state variables in SOC, DOC, MBC, and EEC.
    SOC, DOC, MBC, EEC =  torch.chunk(C_PATH, state_dim, -1) #Partition SOC, DOC, MBC, EEC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor. 
    current_temp = temp_gen(T_SPAN_TENSOR, temp_ref) #Obtain temperature function vector across span of times.
    drift = torch.empty_like(C_PATH, device = C_PATH.device) #Initiate tensor with same dims as C_PATH to assign drift.
    #Decay parameters are forced by temperature changes.
    u_Q = linear_temp_dep(SAWB_C_params_dict['u_Q_ref'], current_temp, SAWB_C_params_dict['Q'], temp_ref) #Apply linear temperature-dependence to u_Q.
    V_D = arrhenius_temp_dep(SAWB_C_params_dict['V_D_ref'], current_temp, SAWB_C_params_dict['Ea_V_D'], temp_ref) #Apply vectorized temperature-dependent transformation to V_D.
    V_U = arrhenius_temp_dep(SAWB_C_params_dict['V_U_ref'], current_temp, SAWB_C_params_dict['Ea_V_U'], temp_ref) #Apply vectorized temperature-dependent transformation to V_U.
    #Drift is calculated.
    drift_SOC = I_S_TENSOR + SAWB_C_params_dict['a_MSA'] * SAWB_C_params_dict['r_M'] * MBC - ((V_D * EEC * SOC) / (SAWB_C_params_dict['K_D'] + SOC))
    drift_DOC = I_D_TENSOR + (1 - SAWB_C_params_dict['a_MSA']) * SAWB_C_params_dict['r_M'] * MBC + ((V_D * EEC * SOC) / (SAWB_C_params_dict['K_D'] + SOC)) + SAWB_C_params_dict['r_L'] * EEC - ((V_U * MBC * DOC) / (SAWB_C_params_dict['K_U'] + DOC))
    drift_MBC = (u_Q * (V_U * MBC * DOC) / (SAWB_C_params_dict['K_U'] + DOC)) - (SAWB_C_params_dict['r_M'] + SAWB_C_params_dict['r_E']) * MBC
    drift_EEC = SAWB_C_params_dict['r_E'] * MBC - SAWB_C_params_dict['r_L'] * EEC
    #CO2 = (1 - u_Q) * (V_U * MBC * DOC) / (SAWB_C_params_dict['K_U'] + DOC)
    #Assign elements to drift vector.
    drift[:, :, 0 : 1] = drift_SOC
    drift[:, :, 1 : 2] = drift_DOC
    drift[:, :, 2 : 3] = drift_MBC
    drift[:, :, 3 : 4] = drift_EEC
    #drift[:, :, 4 : 5] = CO2 #CO2 is not a part of the drift. This is a hack for the explicit algebraic variable situation.
    #Diffusion matrix is assigned.
    diffusion_sqrt_single = torch.diag(torch.sqrt(LowerBound.apply(torch.as_tensor([SAWB_C_params_dict['c_SOC'], SAWB_C_params_dict['c_DOC'], SAWB_C_params_dict['c_MBC'], SAWB_C_params_dict['c_EEC']]), 1e-6))) #Create single diffusion matrix by diagonalizing constant noise scale parameters.    
    #diffusion_sqrt_single = torch.diag(torch.sqrt(LowerBound.apply(torch.as_tensor([SAWB_C_params_dict['c_SOC'], SAWB_C_params_dict['c_DOC'], SAWB_C_params_dict['c_MBC'], SAWB_C_params_dict['c_EEC'], SAWB_C_params_dict['c_CO2']]), 1e-6))) #Create single diffusion matrix by diagonalizing constant noise scale parameters.
    diffusion_sqrt = diffusion_sqrt_single.expand(drift.size(0), drift.size(1), state_dim, state_dim) #Expand diffusion matrices across all paths and across discretized time steps. Diffusion exists for explicit algebraic variable CO2.
    return drift, diffusion_sqrt

#SAWB-ss
def drift_diffusion_SAWB_SS(C_PATH, T_SPAN_TENSOR, I_S_TENSOR, I_D_TENSOR, SAWB_SS_params_dict, temp_gen, temp_ref):
    '''
    Returns SAWB "state scaling diffusion parameterization" drift vectors and diffusion matrices.
    current_temp is output from temp_gen function. 
    Expected SAWB_SS_params_dict = {'u_Q_ref': u_Q_ref, 'Q': Q, 'a_MSA': a_MSA, 'K_D': K_D, 'K_U': K_U, 'V_D_ref': V_D_ref, 'V_U_ref': V_U_ref, 'Ea_V_D': Ea_V_D, 'Ea_V_U': Ea_V_U, 'r_M': r_M, 'r_E': r_E, 'r_L': r_L, 's_SOC': s_SOC, 's_DOC': s_DOC, 's_MBC': s_MBC}
    '''
    state_dim = 4 #SAWB and SAWB-ECA have four state variables in SOC, DOC, MBC, and EEC.
    SOC, DOC, MBC, EEC =  torch.chunk(C_PATH, state_dim, -1) #Partition SOC, DOC, MBC, EEC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor. 
    current_temp = temp_gen(T_SPAN_TENSOR, temp_ref) #Obtain temperature function vector across span of times.
    drift = torch.empty_like(C_PATH, device = C_PATH.device) #Initiate tensor with same dims as C_PATH to assign drift.
    diffusion_sqrt = torch.zeros([drift.size(0), drift.size(1), state_dim, state_dim], device = drift.device) #Create tensor to assign diffusion matrix elements. Diffusion exists for explicit algebraic variable CO2.
    #diffusion_sqrt_diag = torch.empty_like(C_PATH, device = C_PATH.device) #Create tensor to assign diffusion matrix elements.
    #Decay parameters are forced by temperature changes.
    u_Q = linear_temp_dep(SAWB_SS_params_dict['u_Q_ref'], current_temp, SAWB_SS_params_dict['Q'], temp_ref) #Apply linear temperature-dependence to u_Q.
    V_D = arrhenius_temp_dep(SAWB_SS_params_dict['V_D_ref'], current_temp, SAWB_SS_params_dict['Ea_V_D'], temp_ref) #Apply vectorized temperature-dependent transformation to V_D.
    V_U = arrhenius_temp_dep(SAWB_SS_params_dict['V_U_ref'], current_temp, SAWB_SS_params_dict['Ea_V_U'], temp_ref) #Apply vectorized temperature-dependent transformation to V_U.
    #Drift is calculated.
    drift_SOC = I_S_TENSOR + SAWB_SS_params_dict['a_MSA'] * SAWB_SS_params_dict['r_M'] * MBC - ((V_D * EEC * SOC) / (SAWB_SS_params_dict['K_D'] + SOC))
    drift_DOC = I_D_TENSOR + (1 - SAWB_SS_params_dict['a_MSA']) * SAWB_SS_params_dict['r_M'] * MBC + ((V_D * EEC * SOC) / (SAWB_SS_params_dict['K_D'] + SOC)) + SAWB_SS_params_dict['r_L'] * EEC - ((V_U * MBC * DOC) / (SAWB_SS_params_dict['K_U'] + DOC))
    drift_MBC = (u_Q * (V_U * MBC * DOC) / (SAWB_SS_params_dict['K_U'] + DOC)) - (SAWB_SS_params_dict['r_M'] + SAWB_SS_params_dict['r_E']) * MBC
    drift_EEC = SAWB_SS_params_dict['r_E'] * MBC - SAWB_SS_params_dict['r_L'] * EEC
    #CO2 = (1 - u_Q) * (V_U * MBC * DOC) / (SAWB_SS_params_dict['K_U'] + DOC)
    #Assign elements to drift vector.
    drift[:, :, 0 : 1] = drift_SOC
    drift[:, :, 1 : 2] = drift_DOC
    drift[:, :, 2 : 3] = drift_MBC
    drift[:, :, 3 : 4] = drift_EEC
    #drift[:, :, 4 : 5] = CO2 #CO2 is not a part of the drift. This is a hack for the explicit algebraic variable situation.
    #Diffusion matrix is assigned.
    diffusion_sqrt[:, :, 0 : 1, 0] = torch.sqrt(LowerBound.apply(SOC * SAWB_SS_params_dict['s_SOC'], 1e-6)) #SOC diffusion standard deviation
    diffusion_sqrt[:, :, 1 : 2, 1] = torch.sqrt(LowerBound.apply(DOC * SAWB_SS_params_dict['s_DOC'], 1e-6)) #DOC diffusion standard deviation
    diffusion_sqrt[:, :, 2 : 3, 2] = torch.sqrt(LowerBound.apply(MBC * SAWB_SS_params_dict['s_MBC'], 1e-6)) #MBC diffusion standard deviation
    diffusion_sqrt[:, :, 3 : 4, 3] = torch.sqrt(LowerBound.apply(EEC * SAWB_SS_params_dict['s_EEC'], 1e-6)) #EEC diffusion standard deviation
    #diffusion_sqrt[:, :, 4 : 5, 4] = torch.sqrt(LowerBound.apply(CO2 * SAWB_SS_params_dict['s_CO2'], 1e-6)) #CO2 diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 0 : 1] = torch.sqrt(LowerBound.apply(SOC * SAWB_SS_params_dict['s_SOC'], 1e-6)) #SOC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 1 : 2] = torch.sqrt(LowerBound.apply(DOC * SAWB_SS_params_dict['s_DOC'], 1e-6)) #DOC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 2 : 3] = torch.sqrt(LowerBound.apply(MBC * SAWB_SS_params_dict['s_MBC'], 1e-6)) #MBC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 3 : 4] = torch.sqrt(LowerBound.apply(EEC * SAWB_SS_params_dict['s_EEC'], 1e-6)) #EEC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 4 : 5] = torch.sqrt(LowerBound.apply(CO2 * SAWB_SS_params_dict['s_CO2'], 1e-6)) #CO2 diffusion standard deviation
    #diffusion_sqrt = torch.diag_embed(diffusion_sqrt_diag)
    return drift, diffusion_sqrt

#SAWB-ECA-C
def drift_diffusion_SAWB_ECA_C(C_PATH, T_SPAN_TENSOR, I_S_TENSOR, I_D_TENSOR, SAWB_ECA_C_params_dict, temp_gen, temp_ref):
    '''
    Returns SAWB-ECA "constant diffusion parameterization" drift vectors and diffusion matrices.
    current_temp is output from temp_gen function. 
    Expected SAWB_ECA_C_params_dict = {'u_Q_ref': u_Q_ref, 'Q': Q, 'a_MSA': a_MSA, 'K_DE': K_DE, 'K_UE': K_UE, 'V_DE_ref': V_DE_ref, 'V_UE_ref': V_UE_ref, 'Ea_V_DE': Ea_V_DE, 'Ea_V_UE': Ea_V_UE, 'r_M': r_M, 'r_E': r_E, 'r_L': r_L, 'c_SOC': c_SOC, 'c_DOC': c_DOC, 'c_MBC': c_MBC, 'c_EEC': c_EEC}
    '''
    state_dim = 4 #SAWB and SAWB-ECA have four state variables in SOC, DOC, MBC, and EEC.
    SOC, DOC, MBC, EEC =  torch.chunk(C_PATH, state_dim, -1) #Partition SOC, DOC, MBC, EEC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor.
    current_temp = temp_gen(T_SPAN_TENSOR, temp_ref) #Obtain temperature function vector across span of times.
    drift = torch.empty_like(C_PATH, device = C_PATH.device) #Initiate tensor with same dims as C_PATH to assign drift.
    #Decay parameters are forced by temperature changes.
    u_Q = linear_temp_dep(SAWB_ECA_C_params_dict['u_Q_ref'], current_temp, SAWB_ECA_C_params_dict['Q'], temp_ref) #Apply linear temperature-dependence to u_Q.
    V_DE = arrhenius_temp_dep(SAWB_ECA_C_params_dict['V_DE_ref'], current_temp, SAWB_ECA_C_params_dict['Ea_V_DE'], temp_ref) #Apply vectorized temperature-dependent transformation to V_DE.
    V_UE = arrhenius_temp_dep(SAWB_ECA_C_params_dict['V_UE_ref'], current_temp, SAWB_ECA_C_params_dict['Ea_V_UE'], temp_ref) #Apply vectorized temperature-dependent transformation to V_UE.
    #Drift is calculated.
    drift_SOC = I_S_TENSOR + SAWB_ECA_C_params_dict['a_MSA'] * SAWB_ECA_C_params_dict['r_M'] * MBC - ((V_DE * EEC * SOC) / (SAWB_ECA_C_params_dict['K_DE'] + EEC + SOC))
    drift_DOC = I_D_TENSOR + (1 - SAWB_ECA_C_params_dict['a_MSA']) * SAWB_ECA_C_params_dict['r_M'] * MBC + ((V_DE * EEC * SOC) / (SAWB_ECA_C_params_dict['K_DE'] + EEC + SOC)) + SAWB_ECA_C_params_dict['r_L'] * EEC - ((V_UE * MBC * DOC) / (SAWB_ECA_C_params_dict['K_UE'] + MBC + DOC))
    drift_MBC = (u_Q * (V_UE * MBC * DOC) / (SAWB_ECA_C_params_dict['K_UE'] + MBC + DOC)) - (SAWB_ECA_C_params_dict['r_M'] + SAWB_ECA_C_params_dict['r_E']) * MBC
    drift_EEC = SAWB_ECA_C_params_dict['r_E'] * MBC - SAWB_ECA_C_params_dict['r_L'] * EEC
    #CO2 = (1 - u_Q) * (V_UE * MBC * DOC) / (SAWB_ECA_C_params_dict['K_UE'] + MBC + DOC)
    #Assign elements to drift vector.
    drift[:, :, 0 : 1] = drift_SOC
    drift[:, :, 1 : 2] = drift_DOC
    drift[:, :, 2 : 3] = drift_MBC
    drift[:, :, 3 : 4] = drift_EEC
    #drift[:, :, 4 : 5] = CO2 #CO2 is not a part of the drift. This is a hack for the explicit algebraic variable situation.
    #Diffusion matrix is assigned.
    diffusion_sqrt_single = torch.diag(torch.sqrt(LowerBound.apply(torch.as_tensor([SAWB_ECA_C_params_dict['c_SOC'], SAWB_ECA_C_params_dict['c_DOC'], SAWB_ECA_C_params_dict['c_MBC'], SAWB_ECA_C_params_dict['c_EEC']]), 1e-6))) #Create single diffusion matrix by diagonalizing constant noise scale parameters.    
    #diffusion_sqrt_single = torch.diag(torch.sqrt(LowerBound.apply(torch.as_tensor([SAWB_ECA_C_params_dict['c_SOC'], SAWB_ECA_C_params_dict['c_DOC'], SAWB_ECA_C_params_dict['c_MBC'], SAWB_ECA_C_params_dict['c_EEC'], SAWB_ECA_C_params_dict['c_CO2']]), 1e-6))) #Create single diffusion matrix by diagonalizing constant noise scale parameters.
    diffusion_sqrt = diffusion_sqrt_single.expand(drift.size(0), drift.size(1), state_dim, state_dim) #Expand diffusion matrices across all paths and across discretized time steps. Diffusion exists for explicit algebraic variable CO2.
    return drift, diffusion_sqrt

#SAWB-ECA-SS
def drift_diffusion_SAWB_ECA_SS(C_PATH, T_SPAN_TENSOR, I_S_TENSOR, I_D_TENSOR, SAWB_ECA_SS_params_dict, temp_gen, temp_ref):
    '''
    Returns SAWB-ECA "state scaling diffusion parameterization" drift vectors and diffusion matrices.
    current_temp is output from temp_gen function. 
    Expected SAWB_ECA_SS_params_dict = {'u_Q_ref': u_Q_ref, 'Q': Q, 'a_MSA': a_MSA, 'K_DE': K_DE, 'K_UE': K_UE, 'V_DE_ref': V_DE_ref, 'V_UE_ref': V_UE_ref, 'Ea_V_DE': Ea_V_DE, 'Ea_V_UE': Ea_V_UE, 'r_M': r_M, 'r_E': r_E, 'r_L': r_L, 's_SOC': s_SOC, 's_DOC': s_DOC, 's_MBC': s_MBC, 's_EEC': s_EEC}
    '''
    state_dim = 4 #SAWB and SAWB-ECA have four state variables in SOC, DOC, MBC, and EEC.
    SOC, DOC, MBC, EEC =  torch.chunk(C_PATH, state_dim, -1) #Partition SOC, DOC, MBC, EEC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor. 
    current_temp = temp_gen(T_SPAN_TENSOR, temp_ref) #Obtain temperature function vector across span of times.
    drift = torch.empty_like(C_PATH, device = C_PATH.device) #Initiate tensor with same dims as C_PATH to assign drift.
    diffusion_sqrt = torch.zeros([drift.size(0), drift.size(1), state_dim, state_dim], device = drift.device) #Create tensor to assign diffusion matrix elements. Diffusion exists for explicit algebraic variable CO2.
    #diffusion_sqrt_diag = torch.empty_like(C_PATH, device = C_PATH.device) #Create tensor to assign diffusion matrix elements.
    #Decay parameters are forced by temperature changes.
    u_Q = linear_temp_dep(SAWB_ECA_SS_params_dict['u_Q_ref'], current_temp, SAWB_ECA_SS_params_dict['Q'], temp_ref) #Apply linear temperature-dependence to u_Q.
    V_DE = arrhenius_temp_dep(SAWB_ECA_SS_params_dict['V_DE_ref'], current_temp, SAWB_ECA_SS_params_dict['Ea_V_DE'], temp_ref) #Apply vectorized temperature-dependent transformation to V_DE.
    V_UE = arrhenius_temp_dep(SAWB_ECA_SS_params_dict['V_UE_ref'], current_temp, SAWB_ECA_SS_params_dict['Ea_V_UE'], temp_ref) #Apply vectorized temperature-dependent transformation to V_UE.
    #Drift is calculated.
    drift_SOC = I_S_TENSOR + SAWB_ECA_SS_params_dict['a_MSA'] * SAWB_ECA_SS_params_dict['r_M'] * MBC - ((V_DE * EEC * SOC) / (SAWB_ECA_SS_params_dict['K_DE'] + EEC + SOC))
    drift_DOC = I_D_TENSOR + (1 - SAWB_ECA_SS_params_dict['a_MSA']) * SAWB_ECA_SS_params_dict['r_M'] * MBC + ((V_DE * EEC * SOC) / (SAWB_ECA_SS_params_dict['K_DE'] + EEC + SOC)) + SAWB_ECA_SS_params_dict['r_L'] * EEC - ((V_UE * MBC * DOC) / (SAWB_ECA_SS_params_dict['K_UE'] + MBC + DOC))
    drift_MBC = (u_Q * (V_UE * MBC * DOC) / (SAWB_ECA_SS_params_dict['K_UE'] + MBC + DOC)) - (SAWB_ECA_SS_params_dict['r_M'] + SAWB_ECA_SS_params_dict['r_E']) * MBC
    drift_EEC = SAWB_ECA_SS_params_dict['r_E'] * MBC - SAWB_ECA_SS_params_dict['r_L'] * EEC
    #CO2 = (1 - u_Q) * (V_UE * MBC * DOC) / (SAWB_ECA_SS_params_dict['K_UE'] + MBC + DOC)
    #Assign elements to drift vector.
    drift[:, :, 0 : 1] = drift_SOC
    drift[:, :, 1 : 2] = drift_DOC
    drift[:, :, 2 : 3] = drift_MBC
    drift[:, :, 3 : 4] = drift_EEC
    #drift[:, :, 4 : 5] = CO2 #CO2 is not a part of the drift. This is a hack for the explicit algebraic variable situation.
    #Diffusion matrix is assigned.
    diffusion_sqrt[:, :, 0 : 1, 0] = torch.sqrt(LowerBound.apply(SOC * SAWB_ECA_SS_params_dict['s_SOC'], 1e-6)) #SOC diffusion standard deviation
    diffusion_sqrt[:, :, 1 : 2, 1] = torch.sqrt(LowerBound.apply(DOC * SAWB_ECA_SS_params_dict['s_DOC'], 1e-6)) #DOC diffusion standard deviation
    diffusion_sqrt[:, :, 2 : 3, 2] = torch.sqrt(LowerBound.apply(MBC * SAWB_ECA_SS_params_dict['s_MBC'], 1e-6)) #MBC diffusion standard deviation
    diffusion_sqrt[:, :, 3 : 4, 3] = torch.sqrt(LowerBound.apply(EEC * SAWB_ECA_SS_params_dict['s_EEC'], 1e-6)) #EEC diffusion standard deviation
    #diffusion_sqrt[:, :, 4 : 5, 4] = torch.sqrt(LowerBound.apply(CO2 * SAWB_ECA_SS_params_dict['s_CO2'], 1e-6)) #CO2 diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 0 : 1] = torch.sqrt(LowerBound.apply(SOC * SAWB_ECA_SS_params_dict['s_SOC'], 1e-6)) #SOC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 1 : 2] = torch.sqrt(LowerBound.apply(DOC * SAWB_ECA_SS_params_dict['s_DOC'], 1e-6)) #DOC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 2 : 3] = torch.sqrt(LowerBound.apply(MBC * SAWB_ECA_SS_params_dict['s_MBC'], 1e-6)) #MBC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 3 : 4] = torch.sqrt(LowerBound.apply(EEC * SAWB_ECA_SS_params_dict['s_EEC'], 1e-6)) #EEC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 4 : 5] = torch.sqrt(LowerBound.apply(CO2 * SAWB_ECA_SS_params_dict['s_CO2'], 1e-6)) #CO2 diffusion standard deviation
    #diffusion_sqrt = torch.diag_embed(diffusion_sqrt_diag)
    return drift, diffusion_sqrt
