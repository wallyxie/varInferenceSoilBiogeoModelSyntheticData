import torch
torch.set_printoptions(precision=10) #Print additional digits for tensor elements.
import numpy as np

from sbm_temp_functions import *

####################################################
##STOCHASTIC DIFFERENTIAL EQUATION MODEL FUNCTIONS##
####################################################

def litter_drift_and_diffusion_scon(t, batch_size = 3):
    '''
    Returns SCON system exogenous litter input drift. Time t is on hourly scale.
    '''
    I_S = 0.001 + 0.0005 * torch.sin((2 * math.pi / (24 * 365)) * t) #Exogenous SOC input function
    I_D =  0.0001 + 0.00005 * torch.sin((2 * math.pi / (24 * 365)) * t) #Exogenous DOC input function
    I_0 = torch.zeros_like(I_S, device = I_S.device)
    litter_vector = torch.cat([I_S, I_D, I_0], 1)
    litter_vector_perm = litter_vector.permute(0, 2, 1)
    #print('\nlitter_vector size', litter_vector.size())
    #litter_vector_add_dim = litter_vector.repeat(batch_size, 1, 1)
    litter_diffusion_matrix_sqrt = torch.zeros([c_vector.size(0), c_vector.size(2), 3, 3], device=c_vector.device) #Create zeros tensor to assign diffusion matrix elements.
    litter_diffusion_matrix_sqrt[:, :, 0, 0] = torch.sqrt(LowerBound.apply(I_S, 1e-8)).squeeze() #Assigned to element 1, 1 of matrix.
    litter_diffusion_matrix_sqrt[:, :, 1, 1] = torch.sqrt(LowerBound.apply(I_D, 1e-8)).squeeze() #Assigned to element 2, 2 of matrix.
    return litter_vector_perm, litter_diffusion_matrix_sqrt  

def drift_and_diffusion_scon(c_vector, t, scon_params_dict, temp_ref, batch_size = 3):
    '''
    Returns SCON system drift vector and diffusion matrix.
    current_temp is output from temp_gen function. 
    Expected scon_params_dict = {scon_params_dict = {'u_M': u_M, 'a_SD': a_SD, 'a_DS': a_DS, 'a_M': a_M, 'a_MSC': a_MSC, 'k_S_ref': k_S_ref, 'k_D_ref': k_D_ref, 'k_M_ref': k_M_ref, 'Ea_S': Ea_S, 'Ea_D': Ea_D, 'Ea_M': Ea_M}
    '''
    SOC, DOC, MBC = [c_vector[:, i:i + 1, :] for i in range(batch_size)]
    C_vector = SOC, DOC, MBC #Get current system pool density values from C_vector tensor.
    current_temp = temp_gen(t, temp_ref) #Obtain temperature from wave function at current time t.
    #Decay parameters are forced by temperature changes.
    k_S = arrhenius_temp_dep(scon_params_dict['k_S_ref'], current_temp, scon_params_dict['Ea_S'], temp_ref)
    k_D = arrhenius_temp_dep(scon_params_dict['k_D_ref'], current_temp, scon_params_dict['Ea_D'], temp_ref)
    k_M = arrhenius_temp_dep(scon_params_dict['k_M_ref'], current_temp, scon_params_dict['Ea_M'], temp_ref)
    #Drift vector is calculated (without litter input).
    dSOC = scon_params_dict['a_DS'] * k_D * DOC + scon_params_dict['a_M'] * scon_params_dict['a_MSC'] * k_M * MBC - k_S * SOC
    dDOC = scon_params_dict['a_SD'] * k_S * SOC + scon_params_dict['a_M'] * (1 - scon_params_dict['a_MSC']) * k_M * MBC - (scon_params_dict['u_M'] + k_D) * DOC 
    dMBC = scon_params_dict['u_M'] * DOC - k_M * MBC
    drift_vector = torch.cat([dSOC, dDOC, dMBC], 1)
    drift_vector_perm = drift_vector.permute(0, 2, 1)
    #print('\n drift_vector = ', drift_vector)
    #print('\n drift_vector.permute', drift_vector_perm)
    #Diffusion matrix is calculated (recall litter input is not a part of the drift vector or diffusion matrix).
    diffusion_matrix_sqrt = torch.zeros([c_vector.size(0), c_vector.size(2), 3, 3], device=c_vector.device) #Create zeros tensor to assign diffusion matrix elements.
    diffusion_matrix_sqrt[:, :, 0, 0] = torch.sqrt(LowerBound.apply(dSOC, 1e-8)).squeeze() #Assigned to element 1, 1 of matrix.
    diffusion_matrix_sqrt[:, :, 1, 1] = torch.sqrt(LowerBound.apply(dDOC, 1e-8)).squeeze() #Assigned to element 2, 2 of matrix.
    diffusion_matrix_sqrt[:, :, 2, 2] = torch.sqrt(LowerBound.apply(dMBC, 1e-8)).squeeze() #Assigned to element 3, 3 of matrix.
    return drift_vector_perm, diffusion_matrix_sqrt
