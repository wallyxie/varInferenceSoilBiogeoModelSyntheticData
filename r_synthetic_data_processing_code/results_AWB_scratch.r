library(tidyverse)
library(ggthemes)

########################
###Non-param Constants##
########################

temp_ref <- 283

################################
##AWB Family System Parameters##
################################

u_Q_ref <- 0.2
Q <- 0.002
a_MSA <- 0.5
K_D <- 200
K_U <- 1
V_D_ref <- 0.4
V_U_ref <- 0.02
Ea_V_D <- 75
Ea_V_U <- 50
r_M <- 0.0004
r_E <- 0.00001
r_L <- 0.0005

#Separate parameters for full ECA
K_DE <- 200
K_UE <- 1
V_DE_ref <- 0.4
V_UE_ref <- 0.02
Ea_V_DE <- 75
Ea_V_UE <- 50


######################################
##Temperature perturbation functions##
######################################

temp_gen <- function(hour, temp_ref) {
    temp <- temp_ref + hour / (20 * 24 * 365) + 10 * sin((2 * pi / 24) * hour) + 10 * sin((2 * pi / (24 * 365)) * hour)
    return(temp)
}

temp_gen_vec <- Vectorize(temp_gen)

arrhenius_temp_dep <- function(parameter, temp, Ea, temp_ref){
    modified_parameter <- parameter * exp(-Ea / 0.008314 * (1 / temp - 1 / temp_ref))
    return(modified_parameter)
}

arrhenius_temp_dep_vec <- Vectorize(arrhenius_temp_dep)

linear_temp_dep <- function(parameter, temp, Q, temp_ref){
    modified_parameter <- parameter - Q * (temp - temp_ref)
    return(modified_parameter)
}

linear_temp_dep_vec <- Vectorize(linear_temp_dep)

##########################
##Process synthetic data##
##########################

df_temp <- read_csv("synthetic_temp.csv")
hour <- df_temp$hour

temps <- temp_gen_vec(hour, temp_ref)
V_D <- arrhenius_temp_dep_vec(V_D_ref, temps, Ea_V_D, temp_ref)
V_U <- arrhenius_temp_dep_vec(V_U_ref, temps, Ea_V_U, temp_ref)
u_Q <- linear_temp_dep(u_Q_ref, temps, Q, temp_ref)

df_AWB_sol <- read_csv("AWB_synthetic_soln.csv")
SOC <- as.numeric(df_AWB_sol[1,])
DOC <- as.numeric(df_AWB_sol[2,])
MBC <- as.numeric(df_AWB_sol[3,])
EEC <- as.numeric(df_AWB_sol[4,])

CO2 <- (V_U * MBC * DOC) / (K_U + DOC) * (1 - u_Q)

df.AWB <- data.frame(hour, SOC, DOC, MBC, EEC, CO2)

df.AWB.plot.slice <- df.AWB %>% slice(which(row_number() %% 100 == 1)) %>% pivot_longer(!hour, names_to = "soil_pool", values_to = "mg_1_g_soil")
write_csv(df.AWB, "AWB_synthetic_sol_df.csv")

plot.AWB.slice <- ggplot(data = df.AWB.plot.slice, aes(x = hour, y = mg_1_g_soil, colour = soil_pool)) + geom_point() + theme_few()
