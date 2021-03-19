library(tidyverse)
library(ggthemes)

########################
###Non-param Constants##
########################

temp_ref <- 283

#########################
##CON System Parameters##
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

df_temp <- read_csv("synthetic_temp.csv")
hour <- df_temp$hour

temp_vec <- temp_gen_vec(hour, temp_ref)

##########################
##Process synthetic data##
##########################

df_temp <- read_csv("synthetic_temp.csv")
hour <- df_temp$hour

temps <- temp_gen_vec(hour, temp_ref)
k_S <- arrhenius_temp_dep_vec(k_S_ref, temps, Ea_S, temp_ref)
k_D <- arrhenius_temp_dep_vec(k_D_ref, temps, Ea_D, temp_ref) 
k_M <- arrhenius_temp_dep_vec(k_M_ref, temps, Ea_M, temp_ref) 

df_CON_sol <- read_csv("CON_synthetic_soln.csv")
SOC <- as.numeric(df_CON_sol[1,])
DOC <- as.numeric(df_CON_sol[2,])
MBC <- as.numeric(df_CON_sol[3,])

CO2 <- (k_S * SOC * (1 - a_SD)) + (k_D * DOC * (1 - a_DS)) + (k_M * MBC * (1 - a_M)) 

df.CON <- data.frame(hour, SOC, DOC, MBC, CO2)

df.CON.plot.slice <- df.CON %>% slice(which(row_number() %% 100 == 1)) %>% pivot_longer(!hour, names_to = "soil_pool", values_to = "mg_1_g_soil")
write_csv(df.CON, "CON_synthetic_sol_df.csv")

plot.CON.slice <- ggplot(data = df.CON.plot.slice, aes(x = hour, y = mg_1_g_soil, colour = soil_pool)) + geom_point() + theme_few()
