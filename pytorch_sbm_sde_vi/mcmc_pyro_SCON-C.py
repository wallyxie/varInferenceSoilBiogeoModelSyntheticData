import argparse
import os
import time
from pyro.infer import MCMC, NUTS, HMC
from pyro.infer.autoguide.initialization import init_to_sample, init_to_uniform
from mcmc_pyro_utils import model, csv_to_obs_df
from SBM_SDE_classes_minibatch import *
                
def main(args):
    T = 5000
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
    out_filename = '../results/mcmc_SCON-C_very_short2.pt'
    
    # Load data
    N = int(T / dt)
    T_span = torch.linspace(0, T, N + 1)
    obs_ndim = state_dim + 1

    print('Loading data from', obs_file)
    obs_times, obs_vals, obs_errors = csv_to_obs_df(obs_file, obs_ndim, T, obs_error_scale)
    obs_vals = obs_vals.T
    y_dict = {int(t): v for t, v in zip(obs_times, obs_vals)}
    assert y_dict[0].shape == (state_dim + 1, )
    
    # Load hyperparameters of theta
    theta_hyperparams = torch.load(p_theta_file)
    param_names = list(theta_hyperparams.keys())
    theta_hyperparams_list = list(zip(*(theta_hyperparams[k] for k in param_names))) # unzip theta hyperparams from dictionary values into individual lists
    loc_theta, scale_theta, a_theta, b_theta = torch.tensor(theta_hyperparams_list)
    assert loc_theta.shape == scale_theta.shape == a_theta.shape == b_theta.shape == (len(param_names), )

    # Load parameters of x_0
    loc_x0 = torch.load(x0_file)
    scale_x0 = obs_error_scale * loc_x0
    assert loc_x0.shape == scale_x0.shape == (state_dim, )

    # Load parameters of y
    scale_y = obs_errors
    assert obs_errors.shape == (1, state_dim + 1)

    hyperparams = {
        'loc_theta': loc_theta, 
        'scale_theta': scale_theta, 
        'a_theta': a_theta, 
        'b_theta': b_theta, 
        'loc_x0': loc_x0, 
        'scale_x0': scale_x0, 
        'scale_y': scale_y 
    }
    
    # Obtain temperature forcing function.
    temp = temp_gen(T_span, temp_ref, temp_rise)
    
    # Obtain SOC and DOC pool litter input vectors for use in flow SDE functions.
    I_S = i_s(T_span) #Exogenous SOC input function
    I_D = i_d(T_span) #Exogenous DOC input function
    
    # Inference
    torch.manual_seed(args.seed)
    SBM_SDE = model_type(T_span, I_S, I_D, temp, temp_ref, diffusion_type)
    print('Using model', SBM_SDE.__class__.__name__, SBM_SDE.DIFFUSION_TYPE)
    
    kernel = NUTS(model, step_size=args.step_size, adapt_step_size=bool(args.adapt_step_size),
                  init_strategy=init_to_sample, jit_compile=bool(args.jit))
    mcmc = MCMC(kernel,
                num_samples=args.num_samples,
                warmup_steps=args.warmup_steps,
                num_chains=args.num_chains)
    
    # Run MCMC
    start_time = time.process_time()
    mcmc.run(SBM_SDE, T, dt, state_dim, param_names, I_S, I_D, temp,
             obs_times.long(), y=obs_vals, **hyperparams)
    elapsed_time = time.process_time() - start_time
    
    # Save results
    print('Saving MCMC samples and diagnostics to', out_filename)
    samples = mcmc.get_samples(group_by_chain=True)
    diagnostics = mcmc.diagnostics()
    torch.save((samples, diagnostics), out_filename)
    with open(elapsed_time_save_string, 'w') as f:
        print(f'Elapsed time: {elapsed_time}', file = f)

    # Print MCMC diagnostics summary
    mcmc.summary()
    print('Elapsed time:', elapsed_time)

if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-samples", nargs="?", default=200, type=int)
    parser.add_argument("--warmup-steps", nargs="?", default=100, type=int)
    parser.add_argument("--num-chains", nargs="?", default=5, type=int)
    parser.add_argument("--step-size", nargs="?", default=1, type=float)
    parser.add_argument("--adapt-step-size", nargs="?", default=1, type=int)
    parser.add_argument("--jit", nargs="?", default=0, type=int)
    parser.add_argument("--seed", nargs="?", default=1, type=int)
    args = parser.parse_args()

    main(args)
