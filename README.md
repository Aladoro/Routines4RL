# Learning Routines for Effective Off-Policy Reinforcement Learning

This repository contains research code for the *ICML 2021* paper [*Learning Routines for Effective Off-Policy Reinforcement Learning*](https://sites.google.com/view/routines-rl).

# Routine framework

We provide our implementations for the algorithms used to validate the effectiveness of the _routine framework_ on the 
[*DeepMind Control Suite*](https://github.com/deepmind/dm_control), namely: _SAC_, _TD3_, _Routine SAC_, _Routine TD3_, and _FiGAR TD3._

## Requirements

1) To replicate the experiments in this project you need to install the Mujoco
simulation software with a valid license. You can find instructions [here](https://github.com/openai/mujoco-py).

2) The rest of the requirements can be installed with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html),
by utilizing the provided environment file:
```setup
conda env create -f environment.yml
conda activate routine
```

## Replicating the results

We provide the set of parameters used to run the main experiments for the _Routine TD3_ and _Routine SAC_ algorithms in 
the _experiment_params_ folder (in the respective _rttd3_params.json_ and _rtsac_params.json_ configuration files). Each
subfolder contains the parameters for one of the corresponding tasks. 
The parameters for the expressivity analysis are inside the _expressivity_analysis_ subfolder. The parameters to run the 
_TD3_, _SAC_, and _FiGAR-TD3_ baseline algorithms do not vary across environments and are inside the _baseline_params_ 
subfolder (in the respective _td3_params.json_, _sac_params.json_, and _figar_params.json_ configuration files).
To run an experiment, execute _main.py_ specifying the _--path_ of the relevant parameters file, e.g, to run 
the standard version of Routine SAC (L=4), execute:

```setup
python main.py --path experiment_params/cheetah_run_params/rtsac_params.json
```

Additionally, _--directory_path_ can be specified to run experiments with all parameter files inside a directory. _--reps_
can be specified to the experiments for a set number of repetitions. E.g, to run all experiments 5 times, execute:

```setup
python main.py --directory_path experiment_params/ --reps 5
```

To run an experiment with our _FiGAR-TD3_ implementation, include the __--figar__ flag.

## Plotting the results

We provide some basic plotting functionality to reproduce the visualizations in the paper. After obtaining the results,
run _plot.py_ to visualize the returns and number of policy evaluations per episode. 
For example, after running experiments for  the _Cheetah run_ task with the default logging location, the results can be
plotted by executing:

```setup
python plot.py experiments_data/routine_results/cheetah_run/ --smooth 5 --aggregate_envs
```

Similarly, the relative number of policy evaluations per episode can be plotted by running:

```setup
python plot.py experiments_data/routine_results/cheetah_run/ --value episode_trajectory_evals_mean --smooth 5 --yaxis "Policy queries" --aggregate_envs
```

