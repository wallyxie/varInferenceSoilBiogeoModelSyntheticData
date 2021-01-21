import torch
import numpy as np

##########################################################################
##DETERMINISTIC SOIL BIOGEOCHEMICAL MODEL INITIAL STEADY STATE ESTIMATES##
##########################################################################

#Analytical_steady_state_init_awb to be coded later.
def analytical_steady_state_init_con(SOC_input, DOC_input, scon_params_dict):
    '''
    Returns a vector of C pool values to initialize an SCON system corresponding to set of parameter values using the analytical steady state equations of the deterministic CON system.
    Vector elements are in order of S_0, D_0, and M_0.
    Expected scon_params_dict = {scon_params_dict = {'u_M': u_M, 'a_SD': a_SD, 'a_DS': a_DS, 'a_M': a_M, 'a_MSC': a_MSC, 'k_S_ref': k_S_ref, 'k_D_ref': k_D_ref, 'k_M_ref': k_M_ref, 'Ea_S': Ea_S, 'Ea_D': Ea_D, 'Ea_M': Ea_M}    
    '''
    D_0 = (DOC_input + SOC_input * scon_params_dict['a_SD']) / (scon_params_dict['u_M'] + scon_params_dict['k_D_ref'] + scon_params_dict['u_M'] * scon_params_dict['a_M'] * (scon_params_dict['a_MSC'] - 1 - scon_params_dict['a_MSC'] * scon_params_dict['a_SD']) - scon_params_dict['a_DS'] * scon_params_dict['k_D_ref'] * scon_params_dict['a_SD'])
    S_0 = (SOC_input + D_0 * (scon_params_dict['a_DS'] * scon_params_dict['k_D_ref'] + scon_params_dict['u_M'] * scon_params_dict['a_M'] * scon_params_dict['a_MSC'])) / scon_params_dict['k_S_ref']
    M_0 = scon_params_dict['u_M'] * D_0 / scon_params_dict['k_M_ref']
    C_0_vector = torch.reshape(torch.FloatTensor([S_0, D_0, M_0]), [3, 1]) #Create tensor object to store column vector of C0 initial conditions. (Can also create zeros tensor and then assign elements.)
    return C_0_vector
