import torch
import numpy as np

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
    temp = temp_ref + t / (20 * 24 * 365) + 10 * torch.sin((2 * np.pi / 24) * t) + 10 * torch.sin((2 * math.pi / (24 * 365)) * t)
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

def analytical_steady_state_init_con(SOC_input, DOC_input, scon_params_dict):
    '''
    Returns a vector of C pool values to initialize an SCON system corresponding to set of parameter values using the analytical steady state equations of the deterministic CON system.
    Vector elements are in order of S_0, D_0, M_0, and CO2_0.
    Expected scon_params_dict = {'u_M': u_M, 'a_SD': a_SD, 'a_DS': a_DS, 'a_M': a_M, 'a_MSC': a_MSC, 'k_S_ref': k_S_ref, 'k_D_ref': k_D_ref, 'k_M_ref': k_M_ref, 'Ea_S': Ea_S, 'Ea_D': Ea_D, 'Ea_M': Ea_M, '[cs]_SOC': [cs]_SOC, '[cs]_DOC': [cs]_DOC, '[cs]_MBC': [cs]_MBC, '[cs]_CO2': [cs]_CO2}
    '''
    D_0 = (DOC_input + SOC_input * scon_params_dict['a_SD']) / (scon_params_dict['u_M'] + scon_params_dict['k_D_ref'] + scon_params_dict['u_M'] * scon_params_dict['a_M'] * (scon_params_dict['a_MSC'] - 1 - scon_params_dict['a_MSC'] * scon_params_dict['a_SD']) - scon_params_dict['a_DS'] * scon_params_dict['k_D_ref'] * scon_params_dict['a_SD'])
    S_0 = (SOC_input + D_0 * (scon_params_dict['a_DS'] * scon_params_dict['k_D_ref'] + scon_params_dict['u_M'] * scon_params_dict['a_M'] * scon_params_dict['a_MSC'])) / scon_params_dict['k_S_ref']
    M_0 = scon_params_dict['u_M'] * D_0 / scon_params_dict['k_M_ref']
    CO2_0 = scon_params_dict['k_S_ref'] * S_0 * (1 - scon_params_dict['a_SD']) + scon_params_dict['k_D_ref'] * D_0 * (1 - scon_params_dict['a_DS']) + scon_params_dict['k_M_ref'] * M_0 * (1 - scon_params_dict['a_M'])
    C_0_vector = torch.as_tensor([S_0, D_0, M_0, CO2_0])
    return C_0_vector

def analytical_steady_state_init_awb(SOC_input, DOC_input, sawb_params_dict):
    '''
    Returns a vector of C pool values to initialize an SAWB system corresponding to set of parameter values using the analytical steady state equations of the deterministic CON system.
    Vector elements are in order of S_0, D_0, M_0, E_0, and CO2_0.
    Expected sawb_params_dict = {'u_Q_ref': u_Q_ref, 'Q': Q, 'a_MSA': a_MSA, 'K_D': K_D, 'K_U': K_U, 'V_D_ref': V_D_ref, 'V_U_ref': V_U_ref, 'Ea_V_D': Ea_V_D, 'Ea_V_U': Ea_V_U, 'r_M': r_M, 'r_E': r_E, 'r_L': r_L, '[cs]_SOC': [cs]_SOC, '[cs]_DOC': [cs]_DOC, '[cs]_MBC': [cs]_MBC, '[cs]_EEC': [cs]_EEC, '[cs]_CO2': [cs]_CO2}
    '''
    S_0 = -((sawb_params_dict['K_D'] * sawb_params_dict['r_L'] * (SOC_input * sawb_params_dict['r_E'] * (sawb_params_dict['u_Q_ref'] - 1) - sawb_params_dict['a_MSA'] * DOC_input * sawb_params_dict['r_M'] * sawb_params_dict['u_Q_ref'] + SOC_input * sawb_params_dict['r_M'] * (-1 + sawb_params_dict['u_Q_ref'] - sawb_params_dict['a_MSA'] * sawb_params_dict['u_Q_ref']))) / (DOC_input * sawb_params_dict['u_Q_ref'] * (-sawb_params_dict['a_MSA'] * sawb_params_dict['r_L'] * sawb_params_dict['r_M'] + sawb_params_dict['r_E'] * sawb_params_dict['V_D_ref']) + SOC_input * (sawb_params_dict['r_E'] * sawb_params_dict['r_L'] * (sawb_params_dict['u_Q_ref'] - 1) + sawb_params_dict['r_L'] * sawb_params_dict['r_M'] * (-1 + sawb_params_dict['u_Q_ref'] - sawb_params_dict['a_MSA'] * sawb_params_dict['u_Q_ref']) + sawb_params_dict['r_E'] * sawb_params_dict['u_Q_ref'] * sawb_params_dict['V_D_ref'])))
    D_0 = -((sawb_params_dict['K_U'] * (sawb_params_dict['r_E'] + sawb_params_dict['r_M'])) / (sawb_params_dict['r_E'] + sawb_params_dict['r_M'] - sawb_params_dict['u_Q_ref'] * sawb_params_dict['V_U_ref']))
    M_0 = -((SOC_input + DOC_input) * sawb_params_dict['u_Q_ref']) / ((sawb_params_dict['r_E'] + sawb_params_dict['r_M']) * (sawb_params_dict['u_Q_ref'] - 1))
    E_0 = sawb_params_dict['r_E'] * M_0 / sawb_params_dict['r_L']
    #E_0 = -((sawb_params_dict['r_E'] * sawb_params_dict['u_Q_ref'] * (SOC_input + DOC_input)) / (sawb_params_dict['r_L'] * (sawb_params_dict['r_E'] + sawb_params_dict['r_M']) * (sawb_params_dict['u_Q_ref'] - 1)))
    CO2_0 = (1 - sawb_params_dict['u_Q_ref']) * (sawb_params_dict['V_U_ref'] * M_0 * D_0) / (sawb_params_dict['K_U'] + D_0)
    C_0_vector = torch.as_tensor([S_0, D_0, M_0, E_0, CO2_0])
    return C_0_vector

def analytical_steady_state_init_awb_eca(SOC_input, DOC_input, sawb_eca_params_dict):
    '''
    Returns a vector of C pool values to initialize an SAWB-ECA system corresponding to set of parameter values using the analytical steady state equations of the deterministic CON system.
    Vector elements are in order of S_0, D_0, M_0, E_0, and CO2_0.
    Expected sawb_eca_params_dict = {'u_Q_ref': u_Q_ref, 'Q': Q, 'a_MSA': a_MSA, 'K_DE': K_DE, 'K_UE': K_UE, 'V_DE_ref': V_DE_ref, 'V_UE_ref': V_UE_ref, 'Ea_V_DE': Ea_V_DE, 'Ea_V_UE': Ea_V_UE, 'r_M': r_M, 'r_E': r_E, 'r_L': r_L, '[cs]_SOC': [cs]_SOC, '[cs]_DOC': [cs]_DOC, '[cs]_MBC': [cs]_MBC, '[cs]_EEC': [cs]_EEC, '[cs]_CO2': [cs]_CO2}
    '''
    S_0 = ((-sawb_eca_params_dict['K_DE'] * sawb_eca_params_dict['r_L'] * (sawb_eca_params_dict['r_E'] + sawb_eca_params_dict['r_M']) * (sawb_eca_params_dict['u_Q_ref'] - 1) + sawb_eca_params_dict['r_E'] * sawb_eca_params_dict['u_Q_ref'] * (SOC_input + DOC_input)) * (SOC_input * sawb_eca_params_dict['r_E'] * (sawb_eca_params_dict['u_Q_ref'] - 1) - sawb_eca_params_dict['a_MSA'] * DOC_input * sawb_eca_params_dict['r_M'] * sawb_eca_params_dict['u_Q_ref'] + SOC_input * sawb_eca_params_dict['r_M'] * (sawb_eca_params_dict['u_Q_ref'] - sawb_eca_params_dict['a_MSA'] * sawb_eca_params_dict['u_Q_ref'] - 1))) / ((sawb_eca_params_dict['r_E'] + sawb_eca_params_dict['r_M']) * (sawb_eca_params_dict['u_Q_ref'] - 1) * (DOC_input * sawb_eca_params_dict['u_Q_ref'] * (sawb_eca_params_dict['r_E'] * sawb_eca_params_dict['V_DE_ref'] - sawb_eca_params_dict['a_MSA'] * sawb_eca_params_dict['r_L'] * sawb_eca_params_dict['r_M']) + SOC_input * (sawb_eca_params_dict['r_E'] * sawb_eca_params_dict['r_L'] * (sawb_eca_params_dict['u_Q_ref'] - 1) + sawb_eca_params_dict['r_L'] * sawb_eca_params_dict['r_M'] * (sawb_eca_params_dict['u_Q_ref'] - sawb_eca_params_dict['a_MSA'] * sawb_eca_params_dict['u_Q_ref'] - 1) + sawb_eca_params_dict['r_E'] * sawb_eca_params_dict['u_Q_ref'] * sawb_eca_params_dict['V_DE_ref'])))
    D_0 = -(sawb_eca_params_dict['K_UE'] * (sawb_eca_params_dict['r_E'] + sawb_eca_params_dict['r_M']) * (sawb_eca_params_dict['u_Q_ref'] - 1) - (SOC_input + DOC_input) * sawb_eca_params_dict['u_Q_ref']) / ((sawb_eca_params_dict['u_Q_ref'] - 1) * (sawb_eca_params_dict['r_E'] + sawb_eca_params_dict['r_M'] - sawb_eca_params_dict['u_Q_ref'] * sawb_eca_params_dict['V_UE_ref']))
    M_0 = -((SOC_input + DOC_input) * sawb_eca_params_dict['u_Q_ref']) / ((sawb_eca_params_dict['r_E'] + sawb_eca_params_dict['r_M']) * (sawb_eca_params_dict['u_Q_ref'] - 1))
    E_0 = sawb_eca_params_dict['r_E'] * M_0 / sawb_eca_params_dict['r_L']
    #E_0 = -((sawb_params_dict['r_E'] * sawb_params_dict['u_Q_ref'] * (SOC_input + DOC_input)) / (sawb_params_dict['r_L'] * (sawb_params_dict['r_E'] + sawb_params_dict['r_M']) * (sawb_params_dict['u_Q_ref'] - 1)))
    CO2_0 = (1 - sawb_eca_params_dict['u_Q_ref']) * sawb_eca_params_dict['V_UE_ref'] * M_0 * D_0 / (sawb_eca_params_dict['K_UE'] + M_0 + D_0)
    C_0_vector = torch.as_tensor([S_0, D_0, M_0, E_0, CO2_0])
    return C_0_vector

####################################################
##STOCHASTIC DIFFERENTIAL EQUATION MODEL FUNCTIONS##
#################################################### 

#SCON-c
def drift_diffusion_scon_c(C_path, T_span_tensor, I_S_tensor, I_D_tensor, scon_c_params_dict, temp_ref):
    '''
    Returns SCON "constant diffusion parameterization" drift vectors and diffusion matrices.
    current_temp is output from temp_gen function. 
    Expected scon_c_params_dict = {'u_M': u_M, 'a_SD': a_SD, 'a_DS': a_DS, 'a_M': a_M, 'a_MSC': a_MSC, 'k_S_ref': k_S_ref, 'k_D_ref': k_D_ref, 'k_M_ref': k_M_ref, 'Ea_S': Ea_S, 'Ea_D': Ea_D, 'Ea_M': Ea_M, 'c_SOC': c_SOC, 'c_DOC': c_DOC, 'c_MBC': c_MBC, 'c_CO2': c_CO2}
    '''
    state_dim = 4 #SCON will correspond to four sets of observations with inclusion of CO2 alongside state observations.
    SOC, DOC, MBC, CO2 =  torch.chunk(C_path, state_dim, -1) #Partition SOC, DOC, MBC, and CO2 values. Split based on final C_path dim, which specifies state variables and is also indexed as dim #2 in tensor. 
    current_temp = temp_gen(T_span_tensor, temp_ref) #Obtain temperature function vector across span of times.
    drift = torch.empty_like(C_path, device = C_path.device) #Initiate tensor with same dims as C_path to assign drift.
    #Decay parameters are forced by temperature changes.
    k_S = arrhenius_temp_dep(scon_c_params_dict['k_S_ref'], current_temp, scon_c_params_dict['Ea_S'], temp_ref) #Apply vectorized temperature-dependent transformation to k_S_ref.
    k_D = arrhenius_temp_dep(scon_c_params_dict['k_D_ref'], current_temp, scon_c_params_dict['Ea_D'], temp_ref) #Apply vectorized temperature-dependent transformation to k_D_ref.
    k_M = arrhenius_temp_dep(scon_c_params_dict['k_M_ref'], current_temp, scon_c_params_dict['Ea_M'], temp_ref) #Apply vectorized temperature-dependent transformation to k_M_ref.
    #Drift is calculated.
    drift_SOC = I_S_tensor + scon_c_params_dict['a_DS'] * k_D * DOC + scon_c_params_dict['a_M'] * scon_c_params_dict['a_MSC'] * k_M * MBC - k_S * SOC
    drift_DOC = I_D_tensor + scon_c_params_dict['a_SD'] * k_S * SOC + scon_c_params_dict['a_M'] * (1 - scon_c_params_dict['a_MSC']) * k_M * MBC - (scon_c_params_dict['u_M'] + k_D) * DOC
    drift_MBC = scon_c_params_dict['u_M'] * DOC - k_M * MBC
    CO2 = (k_S * SOC * (1 - scon_c_params_dict['a_SD'])) + (k_D * DOC * (1 - scon_c_params_dict['a_DS'])) + (k_M * MBC * (1 - scon_c_params_dict['a_M'])) 
    #Assign elements to drift vector.
    drift[:, :, 0 : 1] = drift_SOC
    drift[:, :, 1 : 2] = drift_DOC
    drift[:, :, 2 : 3] = drift_MBC
    drift[:, :, 3 : 4] = CO2 #CO2 is not a part of the drift. This is a hack for the explicit algebraic variable situation.
    #Diffusion matrix is assigned.
    diffusion_sqrt_single = torch.diag(torch.sqrt(LowerBound.apply(torch.as_tensor([scon_c_params_dict['c_SOC'], scon_c_params_dict['c_DOC'], scon_c_params_dict['c_MBC'], scon_c_params_dict['c_CO2']]), 1e-9))) #Create single diffusion matrix by diagonalizing constant noise scale parameters.
    diffusion_sqrt = diffusion_sqrt_single.expand(drift.size(0), drift.size(1), state_dim, state_dim) #Expand diffusion matrices across all paths and across discretized time steps. Diffusion exists for explicit algebraic variable CO2.
    return drift, diffusion_sqrt

#SCON-ss
def drift_diffusion_scon_ss(C_path, T_span_tensor, I_S_tensor, I_D_tensor, scon_ss_params_dict, temp_ref):
    '''
    Returns SCON "state scaling diffusion parameterization" drift vectors and diffusion matrices.
    current_temp is output from temp_gen function. 
    Expected scon_ss_params_dict = {'u_M': u_M, 'a_SD': a_SD, 'a_DS': a_DS, 'a_M': a_M, 'a_MSC': a_MSC, 'k_S_ref': k_S_ref, 'k_D_ref': k_D_ref, 'k_M_ref': k_M_ref, 'Ea_S': Ea_S, 'Ea_D': Ea_D, 'Ea_M': Ea_M, 's_SOC': s_SOC, 's_DOC': s_DOC, 's_MBC': s_MBC, 's_CO2': s_CO2}
    '''
    state_dim = 4 #SCON will correspond to four sets of observations with inclusion of CO2 alongside state observations.
    SOC, DOC, MBC, CO2 =  torch.chunk(C_path, state_dim, -1) #Partition SOC, DOC, MBC, and CO2 values. Split based on final C_path dim, which specifies state variables and is also indexed as dim #2 in tensor. 
    current_temp = temp_gen(T_span_tensor, temp_ref) #Obtain temperature function vector across span of times.
    drift = torch.empty_like(C_path, device = C_path.device) #Initiate tensor with same dims as C_path to assign drift.
    diffusion_sqrt = torch.zeros([drift.size(0), drift.size(1), state_dim, state_dim], device = drift.device) #Create tensor to assign diffusion matrix elements. Diffusion exists for explicit algebraic variable CO2.
    #diffusion_sqrt_diag = torch.empty_like(C_path, device=C_path.device) #Create tensor to assign diffusion matrix elements.
    #Decay parameters are forced by temperature changes.
    k_S = arrhenius_temp_dep(scon_ss_params_dict['k_S_ref'], current_temp, scon_ss_params_dict['Ea_S'], temp_ref) #Apply vectorized temperature-dependent transformation to k_S_ref.
    k_D = arrhenius_temp_dep(scon_ss_params_dict['k_D_ref'], current_temp, scon_ss_params_dict['Ea_D'], temp_ref) #Apply vectorized temperature-dependent transformation to k_D_ref.
    k_M = arrhenius_temp_dep(scon_ss_params_dict['k_M_ref'], current_temp, scon_ss_params_dict['Ea_M'], temp_ref) #Apply vectorized temperature-dependent transformation to k_M_ref.
    #Drift is calculated.
    drift_SOC = I_S_tensor + scon_ss_params_dict['a_DS'] * k_D * DOC + scon_ss_params_dict['a_M'] * scon_ss_params_dict['a_MSC'] * k_M * MBC - k_S * SOC
    drift_DOC = I_D_tensor + scon_ss_params_dict['a_SD'] * k_S * SOC + scon_ss_params_dict['a_M'] * (1 - scon_ss_params_dict['a_MSC']) * k_M * MBC - (scon_ss_params_dict['u_M'] + k_D) * DOC
    drift_MBC = scon_ss_params_dict['u_M'] * DOC - k_M * MBC
    CO2 = (k_S * SOC * (1 - scon_ss_params_dict['a_SD'])) + (k_D * DOC * (1 - scon_ss_params_dict['a_DS'])) + (k_M * MBC * (1 - scon_ss_params_dict['a_M'])) 
    #Assign elements to drift vector.
    drift[:, :, 0 : 1] = drift_SOC
    drift[:, :, 1 : 2] = drift_DOC
    drift[:, :, 2 : 3] = drift_MBC
    drift[:, :, 3 : 4] = CO2 #CO2 is not a part of the drift. This is a hack for the explicit algebraic variable situation.
    #Diffusion matrix is assigned.
    diffusion_sqrt[:, :, 0 : 1, 0] = torch.sqrt(LowerBound.apply(SOC * scon_ss_params_dict['s_SOC'], 1e-9)) #SOC diffusion standard deviation
    diffusion_sqrt[:, :, 1 : 2, 1] = torch.sqrt(LowerBound.apply(DOC * scon_ss_params_dict['s_DOC'], 1e-9)) #DOC diffusion standard deviation
    diffusion_sqrt[:, :, 2 : 3, 2] = torch.sqrt(LowerBound.apply(MBC * scon_ss_params_dict['s_MBC'], 1e-9)) #MBC diffusion standard deviation
    diffusion_sqrt[:, :, 3 : 4, 3] = torch.sqrt(LowerBound.apply(CO2 * scon_ss_params_dict['s_CO2'], 1e-9)) #CO2 diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 0 : 1] = torch.sqrt(LowerBound.apply(SOC * scon_ss_params_dict['s_SOC'], 1e-9)) #SOC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 1 : 2] = torch.sqrt(LowerBound.apply(DOC * scon_ss_params_dict['s_DOC'], 1e-9)) #DOC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 2 : 3] = torch.sqrt(LowerBound.apply(MBC * scon_ss_params_dict['s_MBC'], 1e-9)) #MBC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 3 : 4] = torch.sqrt(LowerBound.apply(CO2 * scon_ss_params_dict['s_CO2'], 1e-9)) #CO2 diffusion standard deviation
    #diffusion_sqrt = torch.diag_embed(diffusion_sqrt_diag)
    return drift, diffusion_sqrt

#SAWB-c
def drift_diffusion_sawb_c(C_path, T_span_tensor, I_S_tensor, I_D_tensor, sawb_c_params_dict, temp_ref):
    '''
    Returns SAWB "constant diffusion parameterization" drift vectors and diffusion matrices.
    current_temp is output from temp_gen function. 
    Expected sawb_c_params_dict = {'u_Q_ref': u_Q_ref, 'Q': Q, 'a_MSA': a_MSA, 'K_D': K_D, 'K_U': K_U, 'V_D_ref': V_D_ref, 'V_U_ref': V_U_ref, 'Ea_V_D': Ea_V_D, 'Ea_V_U': Ea_V_U, 'r_M': r_M, 'r_E': r_E, 'r_L': r_L, 'c_SOC': c_SOC, 'c_DOC': c_DOC, 'c_MBC': c_MBC, 'c_EEC': c_EEC, 'c_CO2': c_CO2}
    '''
    state_dim = 5 #SAWB will correspond to five sets of observations with inclusion of CO2 alongside state observations.
    SOC, DOC, MBC, EEC, CO2 =  torch.chunk(C_path, state_dim, -1) #Partition SOC, DOC, MBC, EEC, and CO2 values. Split based on final C_path dim, which specifies state variables and is also indexed as dim #2 in tensor. 
    current_temp = temp_gen(T_span_tensor, temp_ref) #Obtain temperature function vector across span of times.
    drift = torch.empty_like(C_path, device=C_path.device) #Initiate tensor with same dims as C_path to assign drift.
    #Decay parameters are forced by temperature changes.
    u_Q = linear_temp_dep(sawb_c_params_dict['u_Q_ref'], current_temp, sawb_c_params_dict['Q'], temp_ref) #Apply linear temperature-dependence to u_Q.
    V_D = arrhenius_temp_dep(sawb_c_params_dict['V_D_ref'], current_temp, sawb_c_params_dict['Ea_V_D'], temp_ref) #Apply vectorized temperature-dependent transformation to V_D.
    V_U = arrhenius_temp_dep(sawb_c_params_dict['V_U_ref'], current_temp, sawb_c_params_dict['Ea_V_U'], temp_ref) #Apply vectorized temperature-dependent transformation to V_U.
    #Drift is calculated.
    drift_SOC = I_S_tensor + sawb_c_params_dict['a_MSA'] * sawb_c_params_dict['r_M'] * MBC - ((V_D * EEC * SOC) / (sawb_c_params_dict['K_D'] + SOC))
    drift_DOC = I_D_tensor + (1 - sawb_c_params_dict['a_MSA']) * sawb_c_params_dict['r_M'] * MBC + ((V_D * EEC * SOC) / (sawb_c_params_dict['K_D'] + SOC)) + sawb_c_params_dict['r_L'] * EEC - ((V_U * MBC * DOC) / (sawb_c_params_dict['K_U'] + DOC))
    drift_MBC = (u_Q * (V_U * MBC * DOC) / (sawb_c_params_dict['K_U'] + DOC)) - (sawb_c_params_dict['r_M'] + sawb_c_params_dict['r_E']) * MBC
    drift_EEC = sawb_c_params_dict['r_E'] * MBC - sawb_c_params_dict['r_L'] * EEC
    CO2 = (1 - u_Q) * (V_U * MBC * DOC) / (sawb_c_params_dict['K_U'] + DOC)
    #Assign elements to drift vector.
    drift[:, :, 0 : 1] = drift_SOC
    drift[:, :, 1 : 2] = drift_DOC
    drift[:, :, 2 : 3] = drift_MBC
    drift[:, :, 3 : 4] = drift_EEC
    drift[:, :, 4 : 5] = CO2 #CO2 is not a part of the drift. This is a hack for the explicit algebraic variable situation.
    #Diffusion matrix is assigned.
    diffusion_sqrt_single = torch.diag(torch.sqrt(LowerBound.apply(torch.as_tensor([sawb_c_params_dict['c_SOC'], sawb_c_params_dict['c_DOC'], sawb_c_params_dict['c_MBC'], sawb_c_params_dict['c_EEC'], sawb_c_params_dict['c_CO2']]), 1e-9))) #Create single diffusion matrix by diagonalizing constant noise scale parameters.
    diffusion_sqrt = diffusion_sqrt_single.expand(drift.size(0), drift.size(1), state_dim, state_dim) #Expand diffusion matrices across all paths and across discretized time steps. Diffusion exists for explicit algebraic variable CO2.
    return drift, diffusion_sqrt

#SAWB-ss
def drift_diffusion_sawb_ss(C_path, T_span_tensor, I_S_tensor, I_D_tensor, sawb_ss_params_dict, temp_ref):
    '''
    Returns SAWB "state scaling diffusion parameterization" drift vectors and diffusion matrices.
    current_temp is output from temp_gen function. 
    Expected sawb_ss_params_dict = {'u_Q_ref': u_Q_ref, 'Q': Q, 'a_MSA': a_MSA, 'K_D': K_D, 'K_U': K_U, 'V_D_ref': V_D_ref, 'V_U_ref': V_U_ref, 'Ea_V_D': Ea_V_D, 'Ea_V_U': Ea_V_U, 'r_M': r_M, 'r_E': r_E, 'r_L': r_L, 's_SOC': s_SOC, 's_DOC': s_DOC, 's_MBC': s_MBC, 's_EEC': s_EEC}
    '''
    state_dim = 5 #SAWB will correspond to five sets of observations with inclusion of CO2 alongside state observations.
    SOC, DOC, MBC, EEC, CO2 =  torch.chunk(C_path, state_dim, -1) #Partition SOC, DOC, MBC, EEC, and CO2 values. Split based on final C_path dim, which specifies state variables and is also indexed as dim #2 in tensor. 
    current_temp = temp_gen(T_span_tensor, temp_ref) #Obtain temperature function vector across span of times.
    drift = torch.empty_like(C_path, device=C_path.device) #Initiate tensor with same dims as C_path to assign drift.
    diffusion_sqrt = torch.zeros([drift.size(0), drift.size(1), state_dim, state_dim], device = drift.device) #Create tensor to assign diffusion matrix elements. Diffusion exists for explicit algebraic variable CO2.
    #diffusion_sqrt_diag = torch.empty_like(C_path, device=C_path.device) #Create tensor to assign diffusion matrix elements.
    #Decay parameters are forced by temperature changes.
    u_Q = linear_temp_dep(sawb_ss_params_dict['u_Q_ref'], current_temp, sawb_ss_params_dict['Q'], temp_ref) #Apply linear temperature-dependence to u_Q.
    V_D = arrhenius_temp_dep(sawb_ss_params_dict['V_D_ref'], current_temp, sawb_ss_params_dict['Ea_V_D'], temp_ref) #Apply vectorized temperature-dependent transformation to V_D.
    V_U = arrhenius_temp_dep(sawb_ss_params_dict['V_U_ref'], current_temp, sawb_ss_params_dict['Ea_V_U'], temp_ref) #Apply vectorized temperature-dependent transformation to V_U.
    #Drift is calculated.
    drift_SOC = I_S_tensor + sawb_ss_params_dict['a_MSA'] * sawb_ss_params_dict['r_M'] * MBC - ((V_D * EEC * SOC) / (sawb_ss_params_dict['K_D'] + SOC))
    drift_DOC = I_D_tensor + (1 - sawb_ss_params_dict['a_MSA']) * sawb_ss_params_dict['r_M'] * MBC + ((V_D * EEC * SOC) / (sawb_ss_params_dict['K_D'] + SOC)) + sawb_ss_params_dict['r_L'] * EEC - ((V_U * MBC * DOC) / (sawb_ss_params_dict['K_U'] + DOC))
    drift_MBC = (u_Q * (V_U * MBC * DOC) / (sawb_ss_params_dict['K_U'] + DOC)) - (sawb_ss_params_dict['r_M'] + sawb_ss_params_dict['r_E']) * MBC
    drift_EEC = sawb_ss_params_dict['r_E'] * MBC - sawb_ss_params_dict['r_L'] * EEC
    CO2 = (1 - u_Q) * (V_U * MBC * DOC) / (sawb_ss_params_dict['K_U'] + DOC)
    #Assign elements to drift vector.
    drift[:, :, 0 : 1] = drift_SOC
    drift[:, :, 1 : 2] = drift_DOC
    drift[:, :, 2 : 3] = drift_MBC
    drift[:, :, 3 : 4] = drift_EEC
    drift[:, :, 4 : 5] = CO2 #CO2 is not a part of the drift. This is a hack for the explicit algebraic variable situation.
    #Diffusion matrix is assigned.
    diffusion_sqrt[:, :, 0 : 1, 0] = torch.sqrt(LowerBound.apply(SOC * sawb_ss_params_dict['s_SOC'], 1e-9)) #SOC diffusion standard deviation
    diffusion_sqrt[:, :, 1 : 2, 1] = torch.sqrt(LowerBound.apply(DOC * sawb_ss_params_dict['s_DOC'], 1e-9)) #DOC diffusion standard deviation
    diffusion_sqrt[:, :, 2 : 3, 2] = torch.sqrt(LowerBound.apply(MBC * sawb_ss_params_dict['s_MBC'], 1e-9)) #MBC diffusion standard deviation
    diffusion_sqrt[:, :, 3 : 4, 3] = torch.sqrt(LowerBound.apply(EEC * sawb_ss_params_dict['s_EEC'], 1e-9)) #EEC diffusion standard deviation
    diffusion_sqrt[:, :, 4 : 5, 4] = torch.sqrt(LowerBound.apply(CO2 * sawb_ss_params_dict['s_CO2'], 1e-9)) #CO2 diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 0 : 1] = torch.sqrt(LowerBound.apply(SOC * sawb_ss_params_dict['s_SOC'], 1e-9)) #SOC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 1 : 2] = torch.sqrt(LowerBound.apply(DOC * sawb_ss_params_dict['s_DOC'], 1e-9)) #DOC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 2 : 3] = torch.sqrt(LowerBound.apply(MBC * sawb_ss_params_dict['s_MBC'], 1e-9)) #MBC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 3 : 4] = torch.sqrt(LowerBound.apply(EEC * sawb_ss_params_dict['s_EEC'], 1e-9)) #EEC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 4 : 5] = torch.sqrt(LowerBound.apply(CO2 * sawb_ss_params_dict['s_CO2'], 1e-9)) #CO2 diffusion standard deviation
    #diffusion_sqrt = torch.diag_embed(diffusion_sqrt_diag)
    return drift, diffusion_sqrt

#SAWB-ECA-c
def drift_diffusion_sawb_eca_c(C_path, T_span_tensor, I_S_tensor, I_D_tensor, sawb_eca_c_params_dict, temp_ref):
    '''
    Returns SAWB-ECA "constant diffusion parameterization" drift vectors and diffusion matrices.
    current_temp is output from temp_gen function. 
    Expected sawb_eca_c_params_dict = {'u_Q_ref': u_Q_ref, 'Q': Q, 'a_MSA': a_MSA, 'K_DE': K_DE, 'K_UE': K_UE, 'V_DE_ref': V_DE_ref, 'V_UE_ref': V_UE_ref, 'Ea_V_DE': Ea_V_DE, 'Ea_V_UE': Ea_V_UE, 'r_M': r_M, 'r_E': r_E, 'r_L': r_L, 'c_SOC': c_SOC, 'c_DOC': c_DOC, 'c_MBC': c_MBC, 'c_EEC': c_EEC, 'c_CO2': c_CO2}
    '''
    state_dim = 5 #SAWB and SAWB-ECA will correspond to five sets of observations with inclusion of CO2 alongside state observations.
    SOC, DOC, MBC, EEC, CO2 =  torch.chunk(C_path, state_dim, -1) #Partition SOC, DOC, MBC, EEC, and CO2 values. Split based on final C_path dim, which specifies state variables and is also indexed as dim #2 in tensor. 
    current_temp = temp_gen(T_span_tensor, temp_ref) #Obtain temperature function vector across span of times.
    drift = torch.empty_like(C_path, device=C_path.device) #Initiate tensor with same dims as C_path to assign drift.
    #Decay parameters are forced by temperature changes.
    u_Q = linear_temp_dep(sawb_eca_c_params_dict['u_Q_ref'], current_temp, sawb_eca_c_params_dict['Q'], temp_ref) #Apply linear temperature-dependence to u_Q.
    V_DE = arrhenius_temp_dep(sawb_eca_c_params_dict['V_DE_ref'], current_temp, sawb_eca_c_params_dict['Ea_V_DE'], temp_ref) #Apply vectorized temperature-dependent transformation to V_DE.
    V_UE = arrhenius_temp_dep(sawb_eca_c_params_dict['V_UE_ref'], current_temp, sawb_eca_c_params_dict['Ea_V_UE'], temp_ref) #Apply vectorized temperature-dependent transformation to V_UE.
    #Drift is calculated.
    drift_SOC = I_S_tensor + sawb_eca_c_params_dict['a_MSA'] * sawb_eca_c_params_dict['r_M'] * MBC - ((V_DE * EEC * SOC) / (sawb_eca_c_params_dict['K_DE'] + EEC + SOC))
    drift_DOC = I_D_tensor + (1 - sawb_eca_c_params_dict['a_MSA']) * sawb_eca_c_params_dict['r_M'] * MBC + ((V_DE * EEC * SOC) / (sawb_eca_c_params_dict['K_DE'] + EEC + SOC)) + sawb_eca_c_params_dict['r_L'] * EEC - ((V_UE * MBC * DOC) / (sawb_eca_c_params_dict['K_UE'] + MBC + DOC))
    drift_MBC = (u_Q * (V_UE * MBC * DOC) / (sawb_eca_c_params_dict['K_UE'] + MBC + DOC)) - (sawb_eca_c_params_dict['r_M'] + sawb_eca_c_params_dict['r_E']) * MBC
    drift_EEC = sawb_eca_c_params_dict['r_E'] * MBC - sawb_eca_c_params_dict['r_L'] * EEC
    CO2 = (1 - u_Q) * (V_UE * MBC * DOC) / (sawb_eca_c_params_dict['K_UE'] + MBC + DOC)
    #Assign elements to drift vector.
    drift[:, :, 0 : 1] = drift_SOC
    drift[:, :, 1 : 2] = drift_DOC
    drift[:, :, 2 : 3] = drift_MBC
    drift[:, :, 3 : 4] = drift_EEC
    drift[:, :, 4 : 5] = CO2 #CO2 is not a part of the drift. This is a hack for the explicit algebraic variable situation.
    #Diffusion matrix is assigned.
    diffusion_sqrt_single = torch.diag(torch.sqrt(LowerBound.apply(torch.as_tensor([sawb_eca_c_params_dict['c_SOC'], sawb_eca_c_params_dict['c_DOC'], sawb_eca_c_params_dict['c_MBC'], sawb_eca_c_params_dict['c_EEC'], sawb_eca_c_params_dict['c_CO2']]), 1e-9))) #Create single diffusion matrix by diagonalizing constant noise scale parameters.
    diffusion_sqrt = diffusion_sqrt_single.expand(drift.size(0), drift.size(1), state_dim, state_dim) #Expand diffusion matrices across all paths and across discretized time steps. Diffusion exists for explicit algebraic variable CO2.
    return drift, diffusion_sqrt

#SAWB-ECA-ss
def drift_diffusion_sawb_eca_ss(C_path, T_span_tensor, I_S_tensor, I_D_tensor, sawb_eca_ss_params_dict, temp_ref):
    '''
    Returns SAWB-ECA "state scaling diffusion parameterization" drift vectors and diffusion matrices.
    current_temp is output from temp_gen function. 
    Expected sawb_eca_ss_params_dict = {'u_Q_ref': u_Q_ref, 'Q': Q, 'a_MSA': a_MSA, 'K_DE': K_DE, 'K_UE': K_UE, 'V_DE_ref': V_DE_ref, 'V_UE_ref': V_UE_ref, 'Ea_V_DE': Ea_V_DE, 'Ea_V_UE': Ea_V_UE, 'r_M': r_M, 'r_E': r_E, 'r_L': r_L, 's_SOC': s_SOC, 's_DOC': s_DOC, 's_MBC': s_MBC, 's_EEC': s_EEC, 's_CO2': s_CO2}
    '''
    state_dim = 5 #SAWB and SAWB-ECA will correspond to five sets of observations with inclusion of CO2 alongside state observations.
    SOC, DOC, MBC, EEC, CO2 =  torch.chunk(C_path, state_dim, -1) #Partition SOC, DOC, MBC, EEC, and CO2 values. Split based on final C_path dim, which specifies state variables and is also indexed as dim #2 in tensor. 
    current_temp = temp_gen(T_span_tensor, temp_ref) #Obtain temperature function vector across span of times.
    drift = torch.empty_like(C_path, device=C_path.device) #Initiate tensor with same dims as C_path to assign drift.
    diffusion_sqrt = torch.zeros([drift.size(0), drift.size(1), state_dim, state_dim], device = drift.device) #Create tensor to assign diffusion matrix elements. Diffusion exists for explicit algebraic variable CO2.
    #diffusion_sqrt_diag = torch.empty_like(C_path, device=C_path.device) #Create tensor to assign diffusion matrix elements.
    #Decay parameters are forced by temperature changes.
    u_Q = linear_temp_dep(sawb_eca_ss_params_dict['u_Q_ref'], current_temp, sawb_eca_ss_params_dict['Q'], temp_ref) #Apply linear temperature-dependence to u_Q.
    V_DE = arrhenius_temp_dep(sawb_eca_ss_params_dict['V_DE_ref'], current_temp, sawb_eca_ss_params_dict['Ea_V_DE'], temp_ref) #Apply vectorized temperature-dependent transformation to V_DE.
    V_UE = arrhenius_temp_dep(sawb_eca_ss_params_dict['V_UE_ref'], current_temp, sawb_eca_ss_params_dict['Ea_V_UE'], temp_ref) #Apply vectorized temperature-dependent transformation to V_UE.
    #Drift is calculated.
    drift_SOC = I_S_tensor + sawb_eca_ss_params_dict['a_MSA'] * sawb_eca_ss_params_dict['r_M'] * MBC - ((V_DE * EEC * SOC) / (sawb_eca_ss_params_dict['K_DE'] + EEC + SOC))
    drift_DOC = I_D_tensor + (1 - sawb_eca_ss_params_dict['a_MSA']) * sawb_eca_ss_params_dict['r_M'] * MBC + ((V_DE * EEC * SOC) / (sawb_eca_ss_params_dict['K_DE'] + EEC + SOC)) + sawb_eca_ss_params_dict['r_L'] * EEC - ((V_UE * MBC * DOC) / (sawb_eca_ss_params_dict['K_UE'] + MBC + DOC))
    drift_MBC = (u_Q * (V_UE * MBC * DOC) / (sawb_eca_ss_params_dict['K_UE'] + MBC + DOC)) - (sawb_eca_ss_params_dict['r_M'] + sawb_eca_ss_params_dict['r_E']) * MBC
    drift_EEC = sawb_eca_ss_params_dict['r_E'] * MBC - sawb_eca_ss_params_dict['r_L'] * EEC
    CO2 = (1 - u_Q) * (V_UE * MBC * DOC) / (sawb_eca_ss_params_dict['K_UE'] + MBC + DOC)
    #Assign elements to drift vector.
    drift[:, :, 0 : 1] = drift_SOC
    drift[:, :, 1 : 2] = drift_DOC
    drift[:, :, 2 : 3] = drift_MBC
    drift[:, :, 3 : 4] = drift_EEC
    drift[:, :, 4 : 5] = CO2 #CO2 is not a part of the drift. This is a hack for the explicit algebraic variable situation.
    #Diffusion matrix is assigned.
    diffusion_sqrt[:, :, 0 : 1, 0] = torch.sqrt(LowerBound.apply(SOC * sawb_eca_ss_params_dict['s_SOC'], 1e-9)) #SOC diffusion standard deviation
    diffusion_sqrt[:, :, 1 : 2, 1] = torch.sqrt(LowerBound.apply(DOC * sawb_eca_ss_params_dict['s_DOC'], 1e-9)) #DOC diffusion standard deviation
    diffusion_sqrt[:, :, 2 : 3, 2] = torch.sqrt(LowerBound.apply(MBC * sawb_eca_ss_params_dict['s_MBC'], 1e-9)) #MBC diffusion standard deviation
    diffusion_sqrt[:, :, 3 : 4, 3] = torch.sqrt(LowerBound.apply(EEC * sawb_eca_ss_params_dict['s_EEC'], 1e-9)) #EEC diffusion standard deviation
    diffusion_sqrt[:, :, 4 : 5, 4] = torch.sqrt(LowerBound.apply(CO2 * sawb_eca_ss_params_dict['s_CO2'], 1e-9)) #CO2 diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 0 : 1] = torch.sqrt(LowerBound.apply(SOC * sawb_eca_ss_params_dict['s_SOC'], 1e-9)) #SOC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 1 : 2] = torch.sqrt(LowerBound.apply(DOC * sawb_eca_ss_params_dict['s_DOC'], 1e-9)) #DOC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 2 : 3] = torch.sqrt(LowerBound.apply(MBC * sawb_eca_ss_params_dict['s_MBC'], 1e-9)) #MBC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 3 : 4] = torch.sqrt(LowerBound.apply(EEC * sawb_eca_ss_params_dict['s_EEC'], 1e-9)) #EEC diffusion standard deviation
    #diffusion_sqrt_diag[:, :, 4 : 5] = torch.sqrt(LowerBound.apply(CO2 * sawb_eca_ss_params_dict['s_CO2'], 1e-9)) #CO2 diffusion standard deviation
    #diffusion_sqrt = torch.diag_embed(diffusion_sqrt_diag)
    return drift, diffusion_sqrt
