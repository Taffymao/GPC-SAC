# PBRL

## Introduction
This is the code for reproducing the results of the paper \[Grid-Mapping Pseudo-Count Constraint for Offline Reinforcement Learning [http://arxiv.org/abs/2404.02545]

This code builds up from \[lifelong-learning] [https://github.com/kzl/lifelong_rl], originally derived from \[rlkit][https://github.com/vitchyr/rlkit]. 
## Prerequisites
Python 3.7 with pytorch 1.7.1

mujoco 2.1.0

d4rl[https://github.com/Farama-Foundation/D4RL]

## Installation and Usage

Here is an example of how to install all the dependencies on Ubuntu:
```bash
conda create -n GPC-SAC python=3.7
conda activate GPC-SAC
cd GPC-SAC-master
pip install -r requirements.txt
git clone [https://github.com/Farama-Foundation/D4RL.git]
cd d4rl
# Remove lines including 'dm_control' in setup.py
pip install -e .
```

## Reproducing the results

For running GPC-SAC on diffent environments, run:
```bash
python -m scripts.gpc_sac --env_name [ENVIRONMENT] --seed [K]  --action_n [N]  --state_n [N]  --beta [I]
```
For example to reproduce the GPC-SAC results for halfcheetah-medium-v2, run:
```bash
python -m scripts.gpc_sac --env_name halfcheetah-medium-v2 --seed 0 --action_n 7 --state_n 7 --beta 2
```
## Execution

The core implementation is given in `GPC-SAC-main\lifelong_rl/trainers/q_learning/gpc-sac`, some other changes are in `GPC-SAC-main\data_management/replay_buffers/mujoco_replay_buffer` and `GPC-SAC-main\data_management/replay_buffers/env_replay_buffer`.

In case of any questions, bugs, suggestions or improvements, please feel free to open an issue.
