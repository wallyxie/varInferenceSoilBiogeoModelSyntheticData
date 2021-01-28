import torch
torch.set_printoptions(precision=10) #Print additional digits for tensor elements.
import numpy as np

from sbm_temp_functions import *

####################################################
##STOCHASTIC DIFFERENTIAL EQUATION MODEL FUNCTIONS##
####################################################

def litter_scon(t):
    '''
    Returns SCON system exogenous litter input vector. Time t is on hourly scale.
    '''
    I_S = 0.001 + 0.0005 * np.sin((2 * np.pi / (24 * 365)) * t) #Exogenous SOC input function
    I_D =  0.0001 + 0.00005 * np.sin((2 * np.pi / (24 * 365)) * t) #Exogenous DOC input function
    litter_vector = torch.reshape(torch.FloatTensor([I_S, I_D, 0]), [3, 1]) #Create tensor object to store column vector of litter elements at time t. No litter input into MBC pool.
    return litter_vector

def drift_scon(c_vector, t, scon_params_dict, temp_ref):
    '''
    Returns SCON system drift vector for approximate p(x).
    current_temp is output from temp_gen function. 
    c_vector[0] is expected to be SOC, c_vector[1] is expected to be DOC, and c_vector[2] is expected to be MBC.
    Expected scon_params_dict = {scon_params_dict = {'u_M': u_M, 'a_SD': a_SD, 'a_DS': a_DS, 'a_M': a_M, 'a_MSC': a_MSC, 'k_S_ref': k_S_ref, 'k_D_ref': k_D_ref, 'k_M_ref': k_M_ref, 'Ea_S': Ea_S, 'Ea_D': Ea_D, 'Ea_M': Ea_M}
    '''
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
    drift_vector = torch.reshape(torch.FloatTensor([SOC_drift, DOC_drift, MBC_drift]), [3, 1]) #Create tensor object to store drift column vector at time t in element order of S, D, and M. (Can also create zeros tensor object and then assign elements.)
    return drift_vector

def diffusion_scon(c_vector, t, scon_params_dict, temp_ref):
    '''
    Returns basic SCON system diffusion matrix in which naive diagonalization is used for stochastic conversion rather than Golightly & Wilkinson reaction network conversion.
    current_temp is output from temp_gen function.
    c_vector[0] is expected to be SOC, c_vector[1] is expected to be DOC, and c_vector[2] is expected to be MBC.
    Expected scon_params_dict = {scon_params_dict = {'u_M': u_M, 'a_SD': a_SD, 'a_DS': a_DS, 'a_M': a_M, 'a_MSC': a_MSC, 'k_S_ref': k_S_ref, 'k_D_ref': k_D_ref, 'k_M_ref': k_M_ref, 'Ea_S': Ea_S, 'Ea_D': Ea_D, 'Ea_M': Ea_M}   
    '''
    C_vector = SOC, DOC, MBC #Get current system pool density values from C_vector tensor.
    current_temp = temp_gen(t, temp_ref) #Obtain temperature from wave function at current time t.
    #Decay parameters are forced by temperature changes.    
    k_S = arrhenius_temp_dep(scon_params_dict['k_S_ref'], current_temp, scon_params_dict['Ea_S'], temp_ref)
    k_D = arrhenius_temp_dep(scon_params_dict['k_D_ref'], current_temp, scon_params_dict['Ea_D'], temp_ref)
    k_M = arrhenius_temp_dep(scon_params_dict['k_M_ref'], current_temp, scon_params_dict['Ea_M'], temp_ref)
    #Diffusion matrix is calculated (recall litter input is not a part of the drift vector or diffusion matrix).
    diffusion_matrix_sqrt = torch.zeros(3,3) #Create zeros tensor to assign diffusion matrix elements.
    a11 = scon_params_dict['a_DS'] * k_D * DOC + scon_params_dict['a_M'] * scon_params_dict['a_MSC'] * k_M * MBC - k_S * SOC
    a22 = scon_params_dict['a_SD'] * k_S * SOC + scon_params_dict['a_M'] * (1 - scon_params_dict['a_MSC']) * k_M * MBC - (scon_params_dict['u_M'] + k_D) * DOC
    a33 = scon_params_dict['u_M'] * DOC - k_M * MBC
    diffusion_matrix_sqrt[0,0] = a11
    diffusion_matrix_sqrt[1,1] = a22
    diffusion_matrix_sqrt[2,2] = a33
    #Perform Cholesky decomposition. Beta diffusion matrix is already diagonal in naive ODE-to-SDE conversion, but need to guarantee positive definite matrix in case of negative diagonal element.
    diffusion_matrix_chol_sqrt = torch.cholesky(torch.mm(diffusion_matrix_sqrt, diffusion_matrix_sqrt.t)) #Guarantees positive Cholesky factorization.
    return diffusion_matrix_chol_sqrt
