import torch as tc
import numpy as np

form sbm_temp_functions import *

####################################################
##STOCHASTIC DIFFERENTIAL EQUATION MODEL FUNCTIONS##
####################################################

def litter_scon(t):
    '''
    Returns SCON system exogenous litter input vector. Time t is on hourly scale.
    '''
    I_S = 0.001 + 0.0005 * np.sin((2 * np.pi / (24 * 365)) * t) #Exogenous SOC input function
    I_D =  0.0001 + 0.00005 * np.sin((2 * np.pi / (24 * 365)) * t) #Exogenous DOC input function
    litter_vector = tc.reshape(tc.FloatTensor([I_S, I_D, 0]), [3, 1]) #Arrange into column vector format
    return litter_vector

def drift_scon(c_vector, scon_params_dict, temp_ref, current_temp):
    '''
    Returns SCON system drift vector for approximate p(x).
    current_temp is output from temp_gen function. 
    c_vector[0] is expected to be SOC, c_vector[1] is expected to be DOC, and c_vector[2] is expected to be MBC.
    '''
    c_vector = SOC, DOC, MBC
    #Decay parameters are forced by temperature changes.
    k_S = arrhenius_temp_dep(scon_params_dict['k_S_ref'], current_temp, scon_params_dict['Ea_S'], temp_ref)
    k_D = arrhenius_temp_dep(scon_params_dict['k_D_ref'], current_temp, scon_params_dict['Ea_D'], temp_ref)
    k_M = arrhenius_temp_dep(scon_params_dict['k_M_ref'], current_temp, scon_params_dict['Ea_M'], temp_ref)
    #Drift vector is calculated (without litter input).
    SOC_drift = scon_params_dict['a_DS'] * k_D * DOC + scon_params_dict['a_M'] * scon_params_dict['a_MSC'] * k_M * MBC - k_S * SOC
    DOC_drift = scon_params_dict['a_SD'] * k_S * SOC + scon_params_dict['a_M'] * (1 - scon_params_dict['a_MSC']) * k_M * MBC - (scon_params_dict['u_M'] + k_D) * DOC 
    MBC_drift = scon_params_dict['u_M'] * DOC - k_M * MBC
    drift_vector = tc.reshape(tc.FloatTensor([SOC_drift, DOC_drift, MBC_drift]), [3, 1]) #Arrange into column vector format
    return drift_vector

def diffusion_scon(c_vector, scon_params_dict, temp_ref, current_temp):
    '''
    Returns basic SCON system diffusion matrix in which naive diagonalization is used for stochastic conversion rather than Golightly & Wilkinson reaction network conversion.
    current_temp is output from temp_gen function.
    c_vector[0] is expected to be SOC, c_vector[1] is expected to be DOC, and c_vector[2] is expected to be MBC.    
    '''
    c_vector = SOC, DOC, MBC
    #Decay parameters are forced by temperature changes.    
    k_S = arrhenius_temp_dep(scon_params_dict['k_S_ref'], current_temp, scon_params_dict['Ea_S'], temp_ref)
    k_D = arrhenius_temp_dep(scon_params_dict['k_D_ref'], current_temp, scon_params_dict['Ea_D'], temp_ref)
    k_M = arrhenius_temp_dep(scon_params_dict['k_M_ref'], current_temp, scon_params_dict['Ea_M'], temp_ref)
    #Diffusion matrix is calculated (without litter input).
    diffusion_matrix = tc.zeros(3,3)
    a11 = scon_params_dict['a_DS'] * k_D * DOC + scon_params_dict['a_M'] * scon_params_dict['a_MSC'] * k_M * MBC - k_S * SOC
    a22 = scon_params_dict['a_SD'] * k_S * SOC + scon_params_dict['a_M'] * (1 - scon_params_dict['a_MSC']) * k_M * MBC - (scon_params_dict['u_M'] + k_D) * DOC
    a33 = scon_params_dict['u_M'] * DOC - k_M * MBC
    diffusion_matrix[0,0] = a11
    diffusion_matrix[1,1] = a22
    diffusion_matrix[2,2] = a33
    #Perform Cholesky decomposition. Beta diffusion matrix is already diagonal in naive ODE-to-SDE conversion, but need to guarantee positive definite matrix in case of negative diagonal element.
    diffusion_matrix_chol = tc.cholesky(tc.mm(diffusion_matrix, diffusion_matrix.t)) #Guarantees positive Cholesky factorization.
    return diffusion_matrix_chol
