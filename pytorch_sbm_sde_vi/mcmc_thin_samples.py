import argparse
import os
import torch

"""
Adapted step sizes:
- SCON-C fix theta = 0.0035504691768437624
- SCON-C = 0.00032161441049538553
- SCON-SS = 0.00022522131621371955
"""

def main(args):
	start = int(args.start) # 0
	end = int(args.end)     # 14
	step_size = float(args.step_size)
	out_dir = os.path.join('training_pt_outputs', args.out_dir)

	for i in range(start, end + 1):
        out_file = os.path.join(out_dir, 'out{}.pt'.format(i))
        samples, model, time = torch.load(out_file, map_location='cpu')
        thin = 1 if (i == end) else 10
		torch.save((samples[::thin], step_size, model, time), out_file)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", nargs="?", default=0, type=int)
    parser.add_argument("--end", nargs="?", default=1, type=int)
    parser.add_argument("--step-size", nargs="?", default=0, type=float)
    parser.add_argument("--out-dir", nargs="?", default=None, type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
