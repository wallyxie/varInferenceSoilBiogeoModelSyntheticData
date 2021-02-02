import torch
torch.set_printoptions(precision=10) #Print additional digits for tensor elements.
import numpy as np

from sbm_temp_functions import *

####################################################
##STOCHASTIC DIFFERENTIAL EQUATION MODEL FUNCTIONS##
####################################################

def drift_and_diffusion_scon(drift_vector, T_span_tensor, dt, I_S, I_D, analytical_steady_state_init_con, scon_params_dict, temp_ref, path_count):
    '''
    Returns SCON litter and system drift vectors and diffusion matrices.
    Recall litter drift and diffusion is separate from system drift and diffusion.
    current_temp is output from temp_gen function. 
    Expected scon_params_dict = {scon_params_dict = {'u_M': u_M, 'a_SD': a_SD, 'a_DS': a_DS, 'a_M': a_M, 'a_MSC': a_MSC, 'k_S_ref': k_S_ref, 'k_D_ref': k_D_ref, 'k_M_ref': k_M_ref, 'Ea_S': Ea_S, 'Ea_D': Ea_D, 'Ea_M': Ea_M}
    '''
    system_batch_size = 3 #SCON and CON will always have three state variables.
    C0 = analytical_steady_state_init_con(I_S[0,0,0].item(), I_D[0,0,0].item(), scon_params_dict)
    print('\n Initial pre-perturbation SOC, DOC, MBC = ', C0)
    drift_vector[:, :, 0] = C0 #Assign deterministically generated initial conditions to all paths.
    diffusion_matrix_sqrt = torch.zeros([drift_vector.size(0), drift_vector.size(2), system_batch_size, system_batch_size], device = drift_vector.device) #Create 3 x 3 zeros tensor to assign diffusion matrix elements.
    diffusion_matrix_sqrt[:, 0, 0, 0] = torch.sqrt(C0[0]) #Assigned S0 to element 1, 1 of matrix.
    diffusion_matrix_sqrt[:, 0, 1, 1] = torch.sqrt(C0[1]) #Assigned D0 to element 2, 2 of matrix.
    diffusion_matrix_sqrt[:, 0, 2, 2] = torch.sqrt(C0[2]) #Assigned M0 to element 3, 3 of matrix.
    current_temp = temp_gen(T_span_tensor, temp_ref) #Obtain temperature from wave function at current time t.
    #Decay parameters are forced by temperature changes.
    k_S = arrhenius_temp_dep(scon_params_dict['k_S_ref'], current_temp, scon_params_dict['Ea_S'], temp_ref)
    k_D = arrhenius_temp_dep(scon_params_dict['k_D_ref'], current_temp, scon_params_dict['Ea_D'], temp_ref)
    k_M = arrhenius_temp_dep(scon_params_dict['k_M_ref'], current_temp, scon_params_dict['Ea_M'], temp_ref)
    #print('\n k_S, k_D, k_M =', [k_S, k_D, k_M])
    #Drift vector is calculated (without litter input).
    for i in range(len(T_span_tensor[0, 0, :]) - 1): #-1 needed in indexing because we are stepping after 0.
        SOC, DOC, MBC = drift_vector[0, :, i]
        SOC += dt * (I_S[0, 0, i] + scon_params_dict['a_DS'] * k_D[0, 0, i] * DOC + scon_params_dict['a_M'] * scon_params_dict['a_MSC'] * k_M[0, 0, i] * MBC - k_S[0, 0, i] * SOC)
        DOC += dt * (I_D[0, 0, i] + scon_params_dict['a_SD'] * k_S[0, 0, i] * SOC + scon_params_dict['a_M'] * (1 - scon_params_dict['a_MSC']) * k_M[0, 0, i] * MBC - (scon_params_dict['u_M'] + k_D[0, 0, i]) * DOC) 
        MBC += dt * (scon_params_dict['u_M'] * DOC - k_M[0, 0, i] * MBC)
        #print('\n SOC, DOC, MBC at', '\t', i, '\t', [SOC, DOC, MBC])
        #Assign to i + 1. What about initial conditions, C0? Are they part of the output vector?
        drift_vector[:, :, i + 1] = torch.as_tensor([SOC, DOC, MBC]) #Assign deterministic means to all paths. CONTINUE HERE.
        #Diffusion matrix is calculated (recall litter input is not a part of the drift vector or diffusion matrix).
        diffusion_matrix_sqrt[:, i + 1, 0, 0] = torch.sqrt(LowerBound.apply(SOC, 1e-8)).squeeze() #Assigned to element 1, 1 of matrix.
        diffusion_matrix_sqrt[:, i + 1, 1, 1] = torch.sqrt(LowerBound.apply(DOC, 1e-8)).squeeze() #Assigned to element 2, 2 of matrix.
        diffusion_matrix_sqrt[:, i + 1, 2, 2] = torch.sqrt(LowerBound.apply(MBC, 1e-8)).squeeze() #Assigned to element 3, 3 of matrix.
    return drift_vector, diffusion_matrix_sqrt
