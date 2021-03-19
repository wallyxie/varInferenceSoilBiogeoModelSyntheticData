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

function analytical_steady_state(SOC_input, DOC_input, u_M, a_SD, a_DS, a_M, a_MSC, k_S_ref, k_D_ref, k_M_ref)
    D₀ = (DOC_input + SOC_input * a_SD) / (u_M + k_D_ref + u_M * a_M * (a_MSC - 1 - a_MSC * a_SD) - a_DS * k_D_ref * a_SD)
    S₀ = (SOC_input + D₀ * (a_DS * k_D_ref + u_M * a_M * a_MSC)) / k_S_ref
    M₀ = u_M * D₀ / k_M_ref
    return [S₀, D₀, M₀]
end

########################
###Non-param Constants##
########################

temp_ref = 283

#########################
##ODE System Parameters##
#########################

u_M = 0.002
a_SD = 0.33
a_DS = 0.33
a_M = 0.33
a_MSC = 0.5
k_S_ref = 0.000025
k_D_ref = 0.005
k_M_ref = 0.0002
Ea_S = 75
Ea_D = 50
Ea_M = 50

###############
##ODE Solving##
###############

tspan = (0., 100000.) #in hours
C₀ = analytical_steady_state(I_S(0), I_D(0), u_M, a_SD, a_DS, a_M, a_MSC, k_S_ref, k_D_ref, k_M_ref)
p = [u_M, a_SD, a_DS, a_M, a_MSC, k_S_ref, k_D_ref, k_M_ref, Ea_S, Ea_D, Ea_M]

function CON!(du, u, p, t)
    S, D, M = u
    u_M, a_SD, a_DS, a_M, a_MSC, k_S_ref, k_D_ref, k_M_ref, Ea_S, Ea_D, Ea_M = p
    du[1] = dS = I_S(t) + a_DS * arrhenius_temp_dep(k_D_ref, temp_gen(t), Ea_D, temp_ref) * D + a_M * a_MSC * arrhenius_temp_dep(k_M_ref, temp_gen(t), Ea_M, temp_ref) * M - arrhenius_temp_dep(k_S_ref, temp_gen(t), Ea_S, temp_ref) * S
    du[2] = dD = I_D(t) + a_SD * arrhenius_temp_dep(k_S_ref, temp_gen(t), Ea_S, temp_ref) * S + a_M * (1 - a_MSC) * arrhenius_temp_dep(k_M_ref, temp_gen(t), Ea_M, temp_ref) * M - u_M * D - arrhenius_temp_dep(k_D_ref, temp_gen(t), Ea_D, temp_ref) * D
    du[3] = dM = u_M * D - arrhenius_temp_dep(k_M_ref, temp_gen(t), Ea_M, temp_ref) * M  
end

CON_prob = ODEProblem(CON!, C₀, tspan, p, save_at = 10) #sample result every 10 hours
CON_sol = solve(CON_prob)

#Debug -- need to test without Arrhenius functions -- Arrhenius functions are the issues -- maybe need to pass more into p?

function CON_no_arr!(du, u, p, t)
    S, D, M = u
    u_M, a_SD, a_DS, a_M, a_MSC, k_S_ref, k_D_ref, k_M_ref, Ea_S, Ea_D, Ea_M = p
    du[1] = dS = I_S(t) + a_DS * k_D_ref * D + a_M * a_MSC * k_M_ref * M - k_S_ref * S
    du[2] = dD = I_D(t) + a_SD * k_S_ref * S + a_M * (1 - a_MSC) * k_M_ref * M - u_M * D - k_D_ref * D
    du[3] = dM = u_M * D - k_M_ref * M  
end

CON_debug_prob = ODEProblem(CON_no_arr!, C₀, tspan, p)
CON_debug_sol = solve(CON_prob_debug)

#Test if function sticks to steady state with no temperature forcing

function CON_ss!(du, u, p, t)
    S, D, M = u
    u_M, a_SD, a_DS, a_M, a_MSC, k_S_ref, k_D_ref, k_M_ref, Ea_S, Ea_D, Ea_M = p
    du[1] = dS = I_S(t) + a_DS * arrhenius_temp_dep(k_D_ref, temp_ref, Ea_D, temp_ref) * D + a_M * a_MSC * arrhenius_temp_dep(k_M_ref, temp_ref, Ea_M, temp_ref) * M - arrhenius_temp_dep(k_S_ref, temp_ref, Ea_S, temp_ref) * S
    du[2] = dD = I_D(t) + a_SD * arrhenius_temp_dep(k_S_ref, temp_ref, Ea_S, temp_ref) * S + a_M * (1 - a_MSC) * arrhenius_temp_dep(k_M_ref, temp_ref, Ea_M, temp_ref) * M - u_M * D - arrhenius_temp_dep(k_D_ref, temp_ref, Ea_D, temp_ref) * D
    du[3] = dM = u_M * D - arrhenius_temp_dep(k_M_ref, temp_ref, Ea_M, temp_ref) * M  
end

CON_ss_prob = ODEProblem(CON_ss!, C₀, tspan, p)
CON_ss_sol = solve(CON_ss_prob)
