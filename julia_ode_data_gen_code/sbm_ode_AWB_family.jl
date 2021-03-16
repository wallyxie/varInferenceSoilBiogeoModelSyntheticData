using DifferentialEquations
using ParameterizedFunctions
using Plots; using PlotlyBase; plotly()

#####################
##Non-ODE Functions##
#####################

function temp_gen(hour, temp_ref)
    temp_ref + hour / (20 * 24 * 365) + 10 * sin((2 * pi / 24) * hour) + 10 * sin((2 * pi / (24 * 365)) * hour)
end

function I_S(hour)
    0.001 + 0.0005 * sin((2 * pi / (24 * 365)) * hour) #Exogenous SOC input
end

function I_D(hour)
    0.0001 + 0.00005 * sin((2 * pi / (24 * 365)) * hour) #Exogenous DOC input
end

function arrhenius_temp_dep(parameter, temp, Ea, temp_ref)
    decayed_parameter = parameter * exp(-Ea / 0.008314 * (1 / temp - 1 / temp_ref))
end

function linear_temp_dep(parameter, temp, Q, temp_ref)
    modified_parameter = parameter - Q * (temp - temp_ref)
end

function analytical_steady_state_AWB_original(SOC_input, DOC_input, u_Q_ref, Q, a_MSA, K_D, K_U, V_D_ref, V_U_ref, Ea_V_D, Ea_V_U, r_M, r_E, r_L, temp_ref)
    u_Q = linear_temp_dep(u_Q_ref, temp_ref, Q, temp_ref)
    V_D = arrhenius_temp_dep(V_D_ref, temp_ref, Ea_V_D, temp_ref)
    V_U = arrhenius_temp_dep(V_U_ref, temp_ref, Ea_V_U, temp_ref)    
    D₀ = -((K_U * (r_E + r_M)) / (r_E + r_M - u_Q * V_U))
    S₀ = -((K_D * r_L * (SOC_input * r_E * (u_Q - 1) - a_MSA * DOC_input * r_M * u_Q + SOC_input * r_M * (-1 + u_Q - a_MSA * u_Q))) / (DOC_input * u_Q * (-a_MSA * r_L * r_M + r_E * V_D) + SOC_input * (r_E * r_L * (u_Q - 1) + r_L * r_M * (-1 + u_Q - a_MSA * u_Q) + r_E * u_Q * V_D)))
    M₀ = -((u_Q * (SOC_input + DOC_input)) / ((r_E + r_M) * (u_Q - 1)))
    E₀ = r_E * M₀ / r_L
    #E₀ = -(((SOC_input + DOC_input) * r_E * r_Q) / (r_L * (r_E + r_M) * (u_Q - 1))) 
    return [S₀, D₀, M₀, E₀]
end

function analytical_steady_state_AWB_full_ECA(SOC_input, DOC_input, u_Q_ref, Q, a_MSA, K_DE, K_UE, V_DE_ref, V_UE_ref, Ea_V_DE, Ea_V_UE, r_M, r_E, r_L, temp_ref)
    u_Q = linear_temp_dep(u_Q_ref, temp_ref, Q, temp_ref)
    V_DE = arrhenius_temp_dep(V_DE_ref, temp_ref, Ea_V_DE, temp_ref)
    V_UE = arrhenius_temp_dep(V_UE_ref, temp_ref, Ea_V_UE, temp_ref)    
    S₀ = ((-K_DE * r_L * (r_E + r_M) * (u_Q - 1) + r_E * u_Q * (SOC_input + DOC_input)) * (SOC_input * r_E * (u_Q - 1) - a_MSA * DOC_input * r_M * u_Q + SOC_input * r_M * (u_Q - a_MSA * u_Q - 1))) / ((r_E + r_M) * (u_Q - 1) * (DOC_input * u_Q * (r_E * V_DE - a_MSA * r_L * r_M) + SOC_input * (r_E * r_L * (u_Q - 1) + r_L * r_M * (u_Q - a_MSA * u_Q - 1) + r_E * u_Q * V_DE)))
    D₀ = -(K_UE * (r_E + r_M) * (u_Q - 1) - (SOC_input + DOC_input) * u_Q) / ((u_Q - 1) * (r_E + r_M - u_Q * V_UE))
    M₀ = -((SOC_input + DOC_input) * u_Q) / ((r_E + r_M) * (u_Q - 1))
    E₀ = r_E * M₀ / r_L
    #E₀ = -(((SOC_input + DOC_input) * r_E * r_Q) / (r_L * (r_E + r_M) * (u_Q - 1)))     
    return [S₀, D₀, M₀, E₀]    
end

########################
###Non-param Constants##
########################

temp_ref = 283

#########################
##ODE System Parameters##
#########################

u_Q_ref = 0.2
Q = 0.002
a_MSA = 0.5
K_D = 200
K_U = 1
V_D_ref = 0.4
V_U_ref = 0.02
Ea_V_D = 75
Ea_V_U = 50
r_M = 0.0004
r_E = 0.00001
r_L = 0.0005

#Separate parameters for full ECA
K_DE = 200
K_UE = 1
V_DE_ref = 0.4
V_UE_ref = 0.02
Ea_V_DE = 75
Ea_V_UE = 50

###############
##ODE Solving##
###############

tspan = (0., 100000.) #in hours
C₀ = analytical_steady_state_AWB_original(I_S(0), I_D(0), u_Q_ref, Q, a_MSA, K_D, K_U, V_D_ref, V_U_ref, Ea_V_D, Ea_V_U, r_M, r_E, r_L, temp_ref)
p = [u_Q_ref, Q, a_MSA, K_D, K_U, V_D_ref, V_U_ref, Ea_V_D, Ea_V_U, r_M, r_E, r_L]

#Test original 2010 AWB version
function AWB!(du, u, p, t)
    S, D, M, E = u
    u_Q_ref, Q, a_MSA, K_D, K_U, V_D_ref, V_U_ref, Ea_V_D, Ea_V_U, r_M, r_E, r_L = p
    du[1] = dS = I_S(t) + a_MSA * r_M * M - ((arrhenius_temp_dep(V_D_ref, temp_gen(t, temp_ref), Ea_V_D, temp_ref) * E * S) / (K_D + S))
    du[2] = dD = I_D(t) + (1 - a_MSA) * r_M * M + ((arrhenius_temp_dep(V_D_ref, temp_gen(t, temp_ref), Ea_V_D, temp_ref) * E * S) / (K_D + S)) + r_L * E - ((arrhenius_temp_dep(V_U_ref, temp_gen(t, temp_ref), Ea_V_U, temp_ref) * M * D) / (K_U + D))
    du[3] = dM = linear_temp_dep(u_Q_ref, temp_gen(t, temp_ref), Q, temp_ref) * ((arrhenius_temp_dep(V_U_ref, temp_gen(t, temp_ref), Ea_V_U, temp_ref) * M * D) / (K_U + D)) - r_M * M - r_E * M
    du[4] = dE = r_E * M - r_L * E
end

AWB_prob = ODEProblem(AWB!, C₀, tspan, p, save_at = 10)
AWB_sol = solve(AWB_prob, dt = 0.05, saveat = 0:10:100000) #sample result every 10 hours, forcing dt = 0.05 hours)

#Test if function sticks to steady state with no temperature forcing
function AWB_ss!(du, u, p, t)
    S, D, M, E = u
    u_Q_ref, Q, a_MSA, K_D, K_U, V_D_ref, V_U_ref, Ea_V_D, Ea_V_U, r_M, r_E, r_L = p
    du[1] = dS = I_S(t) + a_MSA * r_M * M - (V_D_ref * E * S) / (K_D + S)
    du[2] = dD = I_D(t) + (1 - a_MSA) * r_M * M + (V_D_ref * E * S) / (K_D + S) + r_L * E - (V_U_ref * M * D) / (K_U + D)
    du[3] = dM = u_Q_ref * (V_U_ref * M * D) / (K_U + D) - r_M * M - r_E * M
    du[4] = dE = r_E * M - r_L * E
end

AWB_ss_prob = ODEProblem(AWB_ss!, C₀, tspan, p)
AWB_ss_sol = solve(AWB_ss_prob)

function AWB_ss2!(du, u, p, t)
    S, D, M, E = u
    u_Q_ref, Q, a_MSA, K_D, K_U, V_D_ref, V_U_ref, Ea_V_D, Ea_V_U, r_M, r_E, r_L = p
    du[1] = dS = I_S(t) + a_MSA * r_M * M - ((arrhenius_temp_dep(V_D_ref, temp_ref, Ea_V_D, temp_ref) * E * S) / (K_D + S))
    du[2] = dD = I_D(t) + (1 - a_MSA) * r_M * M + ((arrhenius_temp_dep(V_D_ref, temp_ref, Ea_V_D, temp_ref) * E * S) / (K_D + S)) + r_L * E - ((arrhenius_temp_dep(V_U_ref, temp_ref, Ea_V_U, temp_ref) * M * D) / (K_U + D))
    du[3] = dM = linear_temp_dep(u_Q_ref, temp_ref, Q, temp_ref) * ((arrhenius_temp_dep(V_U_ref, temp_ref, Ea_V_U, temp_ref) * M * D) / (K_U + D)) - r_M * M - r_E * M
    du[4] = dE = r_E * M - r_L * E
end

AWB_ss2_prob = ODEProblem(AWB_ss2!, C₀, tspan, p)
AWB_ss2_sol = solve(AWB_ss2_prob)

#Test AWB full ECA version
C₀_ECA = analytical_steady_state_AWB_full_ECA(I_S(0), I_D(0), u_Q_ref, Q, a_MSA, K_DE, K_UE, V_DE_ref, V_UE_ref, Ea_V_DE, Ea_V_UE, r_M, r_E, r_L, temp_ref)
p_ECA = [u_Q_ref, Q, a_MSA, K_DE, K_UE, V_DE_ref, V_UE_ref, Ea_V_DE, Ea_V_UE, r_M, r_E, r_L]

#AWB_ECA version
function AWB_ECA!(du, u, p, t)
    S, D, M, E = u
    u_Q_ref, Q, a_MSA, K_DE, K_UE, V_DE_ref, V_UE_ref, Ea_V_DE, Ea_V_UE, r_M, r_E, r_L = p
    du[1] = dS = I_S(t) + a_MSA * r_M * M - ((arrhenius_temp_dep(V_DE_ref, temp_gen(t, temp_ref), Ea_V_DE, temp_ref) * E * S) / (K_DE + E + S))
    du[2] = dD = I_D(t) + (1 - a_MSA) * r_M * M + ((arrhenius_temp_dep(V_DE_ref, temp_gen(t, temp_ref), Ea_V_DE, temp_ref) * E * S) / (K_DE + E + S)) + r_L * E - ((arrhenius_temp_dep(V_UE_ref, temp_gen(t, temp_ref), Ea_V_UE, temp_ref) * M * D) / (K_UE + M + D))
    du[3] = dM = linear_temp_dep(u_Q_ref, temp_gen(t, temp_ref), Q, temp_ref) * ((arrhenius_temp_dep(V_UE_ref, temp_gen(t, temp_ref), Ea_V_UE, temp_ref) * M * D) / (K_UE + M + D)) - r_M * M - r_E * M
    du[4] = dE = r_E * M - r_L * E
end
