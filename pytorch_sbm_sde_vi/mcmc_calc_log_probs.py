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

def load_samples(filename, fix_theta_dict=None):
    samples, model, time = torch.load(filename, map_location='cpu')
    samples = torch.stack(samples)
    
    if fix_theta_dict is not None:
        logit_theta = None
        x = samples.reshape(-1, model.N, model.state_dim)
    else:
        num_params = len(model.param_names)
        logit_theta = samples[:, :num_params]
        x = samples[:, num_params:].reshape(-1, model.N, model.state_dim)

    return x, logit_theta, model, time

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
    split_terms = int(args.split_terms)

    # SCON-C
    input_dir = 'generated_data/'
    obs_file = os.path.join(input_dir, 'SCON-C_CO2_logit_short_2022_01_20_08_53_sample_y_t_5000_dt_0-01_sd_scale_0-25.csv')
    p_theta_file = os.path.join(input_dir, 'SCON-C_CO2_logit_short_2022_01_20_08_53_sample_y_t_5000_dt_0-01_sd_scale_0-25_hyperparams.pt')
    x0_file = os.path.join(input_dir, 'SCON-C_CO2_logit_short_2022_01_20_08_53_sample_y_t_5000_dt_0-01_sd_scale_0-25_x0_SCON_tensor.pt')
    in_filenames = obs_file, p_theta_file, x0_file
    obs_error_scale = 0.1
    if args.fix_theta_file is not None and args.fix_theta_file != "None":
        fix_theta_file = os.path.join(input_dir, '{}.pt'.format(args.fix_theta_file))
    else:
        fix_theta_file = None
    
    # Load data
    y, _ = load_data(model, obs_error_scale, in_filenames)
    
    # Load fix theta dict (if provided)
    if fix_theta_file:
        fix_theta_dict = {k: torch.tensor(v) for k, v in torch.load(fix_theta_file).items()}
    else:
        fix_theta_dict = None

    # Load MCMC samples
    log_probs = []
    if split_terms:
        log_p_theta = []
        log_p_x = []
        log_p_y = []

    for i in range(outer_iters):
        # Compute joint log prob
        out_file = os.path.join(out_dir, 'out{}.pt'.format(i))
        x, logit_theta, model, time = load_samples(out_file, fix_theta_dict)
        log_probs.append(calc_log_probs(model, x, logit_theta, y, fix_theta_dict))
        
        if split_terms:
            # Compute log p(theta)
            theta = model.p_theta.sigmoid(logit_theta)
            log_p_theta.append(model.p_theta.log_prob(theta)).sum(-1)
    
            # Compute log p(x|theta) and log p(y|x, theta)
            weight_obs_nuts = []
            for x_j, theta_j in zip(x, theta):
                theta_dict = {name: theta_j[name] for name in model.param_names}
                log_p_x_j, weight_obs = model.sde_log_prob(x_j, theta_dict)
                log_p_x.append(log_p_x_j)
                log_p_y.append(model.obs_log_prob(x_j, y, theta_dict, weight_obs))

    log_probs_file = os.path.join(out_dir, 'log_probs.pt')
    log_probs = torch.cat(log_probs)
    
    if split_terms:
        log_p_theta = torch.cat(log_p_theta)
        log_p_x = torch.cat(log_p_x)
        log_p_y = torch.cat(log_p_y)
        
        print('Output shapes:', log_probs.shape, log_p_theta.shape, log_p_x.shape, log_p_y.shape)
        torch.save((log_probs, log_p_theta, log_p_x, log_p_y), log_probs_file)
    
    else:
        print('Output shape:', log_probs.shape)
        torch.save(log_probs, log_probs_file)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-out-files", nargs="?", default=1, type=int)
    parser.add_argument("--out-dir", nargs="?", default=None, type=str)
    parser.add_argument("--split-terms", nargs="?", default=0, type=int)
    parser.add_argument("--fix-theta-file", nargs="?", default=None, type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
    