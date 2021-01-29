import torch
torch.set_printoptions(precision=10) #Print additional digits for tensor elements.
import numpy as np

from sbm_temp_functions import *

####################################################
##STOCHASTIC DIFFERENTIAL EQUATION MODEL FUNCTIONS##
####################################################

def litter_scon(t, batch_size = 3):
    '''
    Returns SCON system exogenous litter input vector. Time t is on hourly scale.
    '''
    I_S = 0.001 + 0.0005 * torch.sin((2 * math.pi / (24 * 365)) * t) #Exogenous SOC input function
    I_D =  0.0001 + 0.00005 * torch.sin((2 * math.pi / (24 * 365)) * t) #Exogenous DOC input function
    I_0 = torch.zeros_like(I_S, device=I_S.device)
    litter_vector = torch.cat([I_S, I_D, I_0], 1)
    return litter_vector.repeat(batch_size, 1, 1)

def drift_scon(c_vector, t, scon_params_dict, temp_ref, batch_size = 3):
    '''
    Returns SCON system drift vector for approximate p(x).
    current_temp is output from temp_gen function. 
    c_vector[0] is expected to be SOC, c_vector[1] is expected to be DOC, and c_vector[2] is expected to be MBC.
    Expected scon_params_dict = {scon_params_dict = {'u_M': u_M, 'a_SD': a_SD, 'a_DS': a_DS, 'a_M': a_M, 'a_MSC': a_MSC, 'k_S_ref': k_S_ref, 'k_D_ref': k_D_ref, 'k_M_ref': k_M_ref, 'Ea_S': Ea_S, 'Ea_D': Ea_D, 'Ea_M': Ea_M}
    '''
    SOC, DOC, MBC = [c_vector[:, i:i+1, :] for i in range(batch_size)]
    C_vector = SOC, DOC, MBC #Get current system pool density values from C_vector tensor.
    current_temp = temp_gen(t, temp_ref) #Obtain temperature from wave function at current time t.
    #Decay parameters are forced by temperature changes.
    k_S = arrhenius_temp_dep(scon_params_dict['k_S_ref'], current_temp, scon_params_dict['Ea_S'], temp_ref)
    k_D = arrhenius_temp_dep(scon_params_dict['k_D_ref'], current_temp, scon_params_dict['Ea_D'], temp_ref)
    k_M = arrhenius_temp_dep(scon_params_dict['k_M_ref'], current_temp, scon_params_dict['Ea_M'], temp_ref)
    #Drift vector is calculated (without litter input).
    SOC_drift = scon_params_dict['a_DS'] * k_D * DOC + scon_params_dict['a_M'] * scon_params_dict['a_MSC'] * k_M * MBC - k_S * SOC
    DOC_drift = scon_params_dict['a_SD'] * k_S * SOC + scon_params_dict['a_M'] * (1 - scon_params_dict['a_MSC']) * k_M * MBC - (scon_params_dict['u_M'] + k_D) * DOC 
    MBC_drift = scon_params_dict['u_M'] * DOC - k_M * MBC
    drift_vector = torch.cat([SOC_drift, DOC_drift, MBC_drift], 1)
    # drift_vector = torch.reshape(torch.FloatTensor([SOC_drift, DOC_drift, MBC_drift]), [3, 1]) #Create tensor object to store drift column vector at time t in element order of S, D, and M. (Can also create zeros tensor object and then assign elements.)
    return drift_vector.permute(0, 2, 1)
    
def diffusion_scon(c_vector, t, scon_params_dict, temp_ref):
  '''
  Returns basic SCON system diffusion matrix in which naive diagonalization is used for stochastic conversion rather than Golightly & Wilkinson reaction network conversion.
  current_temp is output from temp_gen function.
  c_vector[0] is expected to be SOC, c_vector[1] is expected to be DOC, and c_vector[2] is expected to be MBC.
  Expected scon_params_dict = {scon_params_dict = {'u_M': u_M, 'a_SD': a_SD, 'a_DS': a_DS, 'a_M': a_M, 'a_MSC': a_MSC, 'k_S_ref': k_S_ref, 'k_D_ref': k_D_ref, 'k_M_ref': k_M_ref, 'Ea_S': Ea_S, 'Ea_D': Ea_D, 'Ea_M': Ea_M}   
  '''
  SOC, DOC, MBC = [c_vector[:, i:i+1, :] for i in range(3)] #Get current system pool density values from C_vector tensor.
  current_temp = temp_gen(t, temp_ref) #Obtain temperature from wave function at current time t.
  #Decay parameters are forced by temperature changes.    
  k_S = arrhenius_temp_dep(scon_params_dict['k_S_ref'], current_temp, scon_params_dict['Ea_S'], temp_ref)
  k_D = arrhenius_temp_dep(scon_params_dict['k_D_ref'], current_temp, scon_params_dict['Ea_D'], temp_ref)
  k_M = arrhenius_temp_dep(scon_params_dict['k_M_ref'], current_temp, scon_params_dict['Ea_M'], temp_ref)
  #Diffusion matrix is calculated (recall litter input is not a part of the drift vector or diffusion matrix).
  diffusion_matrix_sqrt = torch.zeros([c_vector.size(0), c_vector.size(2), 3,3], device=c_vector.device) #Create zeros tensor to assign diffusion matrix elements.
  a11 = scon_params_dict['a_DS'] * k_D * DOC + scon_params_dict['a_M'] * scon_params_dict['a_MSC'] * k_M * MBC - k_S * SOC
  a22 = scon_params_dict['a_SD'] * k_S * SOC + scon_params_dict['a_M'] * (1 - scon_params_dict['a_MSC']) * k_M * MBC - (scon_params_dict['u_M'] + k_D) * DOC
  a33 = scon_params_dict['u_M'] * DOC - k_M * MBC
  diffusion_matrix_sqrt[:, :, 0, 0] = torch.sqrt(LowerBound.apply(a11, 1e-6)).squeeze()
  diffusion_matrix_sqrt[:, :, 1, 1] = torch.sqrt(LowerBound.apply(a22, 1e-6)).squeeze()
  diffusion_matrix_sqrt[:, :, 2, 2] = torch.sqrt(LowerBound.apply(a33, 1e-6)).squeeze()
  return diffusion_matrix_sqrt
