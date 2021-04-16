import torch
from SBM_SDE import *
from obs_and_flow_classes_and_functions import LowerBound

def get_CO2_CON(C_PATH, T_SPAN_TENSOR, SCON_params_dict, TEMP_GEN, TEMP_REF):
    state_dim = 3 #SCON has three state variables in SOC, DOC, and MBC.
    SOC, DOC, MBC =  torch.chunk(C_PATH, state_dim, -1) #Partition SOC, DOC, and MBC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor.
    current_temp = TEMP_GEN(T_SPAN_TENSOR, TEMP_REF)
    #Decay parameters are forced by temperature changes.    
    k_S = arrhenius_temp_dep(SCON_params_dict['k_S_ref'], current_temp, SCON_params_dict['Ea_S'], TEMP_REF) #Apply vectorized temperature-dependent transformation to k_S_ref.
    k_D = arrhenius_temp_dep(SCON_params_dict['k_D_ref'], current_temp, SCON_params_dict['Ea_D'], TEMP_REF) #Apply vectorized temperature-dependent transformation to k_D_ref.
    k_M = arrhenius_temp_dep(SCON_params_dict['k_M_ref'], current_temp, SCON_params_dict['Ea_M'], TEMP_REF) #Apply vectorized temperature-dependent transformation to k_M_ref.
    CO2 = (k_S * SOC * (1 - SCON_params_dict['a_SD'])) + (k_D * DOC * (1 - SCON_params_dict['a_DS'])) + (k_M * MBC * (1 - SCON_params_dict['a_M']))
    return LowerBound.apply(CO2, 1e-10)

def get_CO2_AWB(C_PATH, T_SPAN_TENSOR, SAWB_params_dict, TEMP_GEN, TEMP_REF):
    state_dim = 4 #SAWB and SAWB-ECA have four state variables in SOC, DOC, MBC, and EEC.
    SOC, DOC, MBC, EEC =  torch.chunk(C_PATH, state_dim, -1) #Partition SOC, DOC, MBC, EEC values. Split based on final C_PATH dim, which specifies state variables and is also indexed as dim #2 in tensor. 
    current_temp = TEMP_GEN(T_SPAN_TENSOR, TEMP_REF)
    #Decay parameters are forced by temperature changes.    
    u_Q = linear_temp_dep(SAWB_C_params_dict['u_Q_ref'], current_temp, SAWB_C_params_dict['Q'], TEMP_REF) #Apply linear temperature-dependence to u_Q.
    V_D = arrhenius_temp_dep(SAWB_C_params_dict['V_D_ref'], current_temp, SAWB_C_params_dict['Ea_V_D'], TEMP_REF) #Apply vectorized temperature-dependent transformation to V_D.
    V_U = arrhenius_temp_dep(SAWB_C_params_dict['V_U_ref'], current_temp, SAWB_C_params_dict['Ea_V_U'], TEMP_REF) #Apply vectorized temperature-dependent transformation to V_U.
    CO2 = (1 - u_Q) * (V_U * MBC * DOC) / (SAWB_C_params_dict['K_U'] + DOC)
    return LowerBound.apply(CO2, 1e-10)
