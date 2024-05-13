import argparse
import os
import torch
from SBM_SDE_classes_mcmc import *

torch.set_default_dtype(torch.float64)

def calc_log_probs(model, x, logit_theta, y, fix_theta_dict=None):
    if fix_theta_dict is not None:
        logit_theta = [None] * len(x)
    model.device = torch.device('cpu')
    return torch.tensor([model.log_prob(x_i, y, logit_theta_i, fix_theta_dict) for x_i, logit_theta_i in zip(x, logit_theta)])

def load_samples(filename, fix_theta=False):
    samples, model, time = torch.load(filename, map_location='cpu')
    samples = torch.stack(samples)
    if fix_theta:
        
        x = samples.reshape(-1, model.N, model.state_dim)
        theta = None
    else:
        num_params = len(model.param_names)
        theta = samples[:, :num_params]
        x = samples[:, num_params:].reshape(-1, model.N, model.state_dim)
    return x, theta, model, time

def load_data(model, obs_error_scale, in_filenames):
    model.device = torch.device('cpu')
    return model.load_data(obs_error_scale, *in_filenames)

def main(args):
    T = 5000
    dt = 1.0
    obs_CO2 = True    # whether or not to use CO2 obs (True to use CO2, False o/w)
    state_dim = 3     # assumes that obs_dim = state_dim + 1, with the last dim being CO2
    obs_error_scale = 0.1
    temp_ref = 283
    temp_rise = 5     # High estimate of 5 celsius temperature rise by 2100. 
    model = SCON(T, dt, state_dim, temp_ref, temp_rise, 'C')

    # Output files
    out_dir = os.path.join('training_pt_outputs', args.out_dir)
    outer_iters = int(args.num_out_files)

    # SCON-C
    input_dir = 'generated_data/'
    obs_file = os.path.join(input_dir, 'SCON-C_CO2_logit_short_2022_01_20_08_53_sample_y_t_5000_dt_0-01_sd_scale_0-25.csv')
    p_theta_file = os.path.join(input_dir, 'SCON-C_CO2_logit_short_2022_01_20_08_53_sample_y_t_5000_dt_0-01_sd_scale_0-25_hyperparams.pt')
    x0_file = os.path.join(input_dir, 'SCON-C_CO2_logit_short_2022_01_20_08_53_sample_y_t_5000_dt_0-01_sd_scale_0-25_x0_SCON_tensor.pt')
    in_filenames = obs_file, p_theta_file, x0_file
    obs_error_scale = 0.1
    
    # Load data
    y, _ = load_data(model, obs_error_scale, in_filenames)

    # Load MCMC samples
    log_probs = []
    for i in range(outer_iters):
        out_file = os.path.join(out_dir, 'out{}.pt'.format(i))
        x, logit_theta, model, time = load_samples(out_file, fix_theta=False)
        log_probs.append(calc_log_probs(model, x, logit_theta, y, None))
    log_probs = torch.cat(log_probs)

    log_probs_file = os.path.join(out_dir, 'log_probs.pt')
    torch.save(log_probs, log_probs_file)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-out-files", nargs="?", default=1, type=int)
    parser.add_argument("--out-dir", nargs="?", default=None, type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
    