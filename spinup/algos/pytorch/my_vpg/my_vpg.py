"""My implementation of VPG.

Modified from my_template.
"""

import time
import numpy as np
import torch
from torch.optim import Adam
import gym
import spinup.algos.pytorch.my_template.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

# TODO create an experience buffer if needed

def my_vpg(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict()):
    """
    TODO implement simple version of algorithm following OpenAI's standard set of steps:
    1) Logger setup
    2) Random seed setting
    3) Environment instantiation
    4) Making placeholders for the computation graph
    5) Building the actor-critic computation graph via the actor_critic function passed to
    the algorithm function as an argument
    6) Instantiating the experience buffer
    7) Building the computation graph for loss functions and diagnostics specific to the
    algorithm
    8) Making training ops
    9) Setting up model saving through the logger
    10) Defining functions needed for running the main loop of the algorithm (e.g. the core
    update function, get action function, and test agent function, depending on the algorithm)
    11) Running the main loop of the algorithm:
        a) Run the agent in the environment
        b) Periodically update the parameters of the agent according to the main equations of
        the algorithm
        c) Log key performance metrics and save agent
    """
    pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='my_vpg')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    my_vpg(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(
            # TODO complete kwargs according to algorithm
        )
    )
