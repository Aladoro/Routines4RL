import argparse
import os
from run_experiment import run_experiment_from_params
from run_figar_experiment import run_experiment_from_params as run_experiment_from_params_figar

parser = argparse.ArgumentParser(description='Routine off-policy RL')

parser.add_argument('--path', default='experiment_params/params.json', help='Path of the parameter file to use.')
parser.add_argument('--directory_path', default=None,
                    help='Path of the directory containing all parameter files to use.')
parser.add_argument('--reps', default=1, help='Number of repetitions for each experiment.')
parser.add_argument('--gpu', default=0, help='GPU to use.')
parser.add_argument('--memory', default=10000, help='Maximum GPU memory to allocate.')
parser.add_argument('--random_seed', action='store_true', help='Resample a different random seed for each experiment.')
parser.add_argument('--figar', action='store_true', help='Run an experiment using FiGAR.')

args = parser.parse_args()

if args.directory_path is None:
    if args.figar:
        run_experiment_from_params_figar(param_path=args.path, repetitions=int(args.reps), gpu=int(args.gpu),
                                         memory_limit=float(args.memory), randomize_seed=args.random_seed)
    else:
        run_experiment_from_params(param_path=args.path, repetitions=int(args.reps), gpu=int(args.gpu),
                                   memory_limit=float(args.memory), randomize_seed=args.random_seed)
else:
    for root, dir, files in os.walk(args.directory_path):
        for file in files:
            param_path = os.path.join(root, file)
            if args.figar:
                run_experiment_from_params_figar(param_path=param_path, repetitions=int(args.reps),
                                                 gpu=int(args.gpu), memory_limit=float(args.memory),
                                                 randomize_seed=args.random_seed)
            else:
                run_experiment_from_params(param_path=param_path, repetitions=int(args.reps),
                                           gpu=int(args.gpu), memory_limit=float(args.memory),
                                           randomize_seed=args.random_seed)
