import os
from pyro.infer import MCMC
from mcmc_utils import parse_args, run
from SBM_SDE_classes_minibatch import *
                
def main(args):
    T = 500
    dt = 1.0
    obs_CO2 = True    # whether or not to use CO2 obs (True to use CO2, False o/w)
    state_dim = 3     # assumes that obs_dim = state_dim + 1, with the last dim being CO2
    obs_error_scale = 0.1
    temp_ref = 283
    temp_rise = 5     # High estimate of 5 celsius temperature rise by 2100. 

    # Inference parameters
    model_type = SCON
    diffusion_type = 'C'

    # Input files
    input_dir = 'generated_data'
    obs_file = os.path.join(input_dir, 'SCON-C_CO2_logit_short_2022_01_20_08_53_sample_y_t_5000_dt_0-01_sd_scale_0-25.csv')
    p_theta_file = os.path.join(input_dir, 'SCON-C_CO2_logit_short_2022_01_20_08_53_sample_y_t_5000_dt_0-01_sd_scale_0-25_hyperparams.pt')
    x0_file = os.path.join(input_dir, 'SCON-C_CO2_logit_short_2022_01_20_08_53_sample_y_t_5000_dt_0-01_sd_scale_0-25_x0_SCON_tensor.pt')

    # Output file
    out_dir = 'training_pt_outputs'
    samples_file = os.path.join(out_dir, 'mcmc_SCON-C_500_samples.pt')
    diagnostics_file = os.path.join(out_dir, 'mcmc_SCON-C_500_diagnostics.pt')
    times_file = os.path.join(out_dir, 'mcmc_SCON-C_500_times.txt')
    args_file = os.path.join(out_dir, 'mcmc_SCON-C_500_args.txt')

    model_params = T, dt, obs_CO2, state_dim, obs_error_scale, \
        temp_ref, temp_rise, model_type, diffusion_type
    in_filenames = obs_file, p_theta_file, x0_file
    out_filenames = out_dir, samples_file, diagnostics_file, times_file, args_file

    run(args, model_params, in_filenames, out_filenames)

if __name__ == "__main__":
    args = parse_args()
    main(args)
