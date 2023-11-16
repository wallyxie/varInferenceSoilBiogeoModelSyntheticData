import argparse
import time
import os
import sys
import torch
torch.set_default_dtype(torch.float64)

import pyro
from pyro.infer import MCMC, NUTS, HMC
from pyro.infer.autoguide.initialization import init_to_sample, init_to_uniform

class Logger:
    def __init__(self, log_dir, num_samples, save_every=10):
        self.save_every = save_every
        self.num_samples = num_samples
        self.log_dir = log_dir

        self.t0 = time.process_time()
        self.total_time = 0
        self.times = []
        self.samples = []
        
    def log(self, kernel, samples, stage, i):
        if stage == 'Sample':
            self.times.append(time.process_time() - self.t0)
            self.total_time = self.times[-1]
            self.samples.append(samples)

            if i % self.save_every == 0 or i == self.num_samples:
                # Save log
                progress_dict = {'kernel': kernel,
                                 'samples': self.samples,
                                 'times': self.times}
                progress_file = '{}/iter{}.pt'.format(self.log_dir, i)
                print('Saving progress to', progress_file)
                sys.stdout.flush()
                torch.save(progress_dict, progress_file)
    
                # Clear log
                self.times = []
                self.samples = []

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-samples", nargs="?", default=200, type=int)
    parser.add_argument("--num-chains", nargs="?", default=5, type=int)
    parser.add_argument("--warmup-steps", nargs="?", default=100, type=int)
    parser.add_argument("--step-size", nargs="?", default=1, type=float)
    parser.add_argument("--adapt-step-size", nargs="?", default=1, type=int)
    parser.add_argument("--jit-compile", nargs="?", default=1, type=int)
    parser.add_argument("--seed", nargs="?", default=1, type=int)
    parser.add_argument("--save-every", nargs="?", default=10, type=int)
    parser.add_argument("--name", nargs="?", default='mcmc', type=str)
    args = parser.parse_args()
    return args

def run(args, model_params, in_filenames, out_filenames):
    T, dt, obs_CO2, state_dim, obs_error_scale, \
        temp_ref, temp_rise, model_type, diffusion_type, device = model_params
    out_dir, samples_file, diagnostics_file, model_file = out_filenames
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Instantiate SCON object
    SBM_SDE = model_type(T, dt, state_dim, temp_ref, temp_rise, diffusion_type) #.to(device)
    y = SBM_SDE.load_data(obs_error_scale, *in_filenames) #.to(device)
    print(y.get_device(), SBM_SDE.temp.get_device())
    print('Using model', SBM_SDE.__class__.__name__, SBM_SDE.diffusion_type)
    mp_context = 'spawn' if device == 'cuda' and args.num_chains > 1 else None
    print(mp_context, device, args.num_chains, device == 'cuda', args.num_chains > 1)

    # Instantiate MCMC object
    kernel = NUTS(SBM_SDE.model,
                  step_size=args.step_size,
                  adapt_step_size=bool(args.adapt_step_size),
                  init_strategy=init_to_sample,
                  jit_compile=bool(args.jit_compile), ignore_jit_warnings=True)
    logger = Logger(out_dir, args.num_samples, args.save_every)
    mcmc = MCMC(kernel,
                num_samples=args.num_samples,
                warmup_steps=args.warmup_steps,
                num_chains=args.num_chains,
                hook_fn=logger.log,
                mp_context=mp_context)
    
    # Run MCMC and record runtime per iteration
    torch.manual_seed(args.seed)
    print('Running MCMC on device', device)
    sys.stdout.flush()
    mcmc.run(y)
    print('Total time:', logger.total_time, 'seconds')
    
    # Save results
    print('Saving MCMC samples and diagnostics to', out_dir)
    samples = mcmc.get_samples(group_by_chain=True)
    diagnostics = mcmc.diagnostics()
    
    torch.save(samples, samples_file)
    torch.save(diagnostics, diagnostics_file)
    torch.save(SBM_SDE, model_file)

    # Print MCMC diagnostics summary
    mcmc.summary()
