"""Standard template algorithm file for all my implementations of algorithms.

This is a standard template of the algorithm implementation to be run similarly to
the OpenAI written algorithms.

This template will be reused across my implementations starting with prefix "my_" as
reimplement algorithms after reading their original papers.
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

def my_algorithm(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict()seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, pi_lr=3e-4,
        max_ep_len=1000, logger_kwargs=dict(), save_freq=10):
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

    """ OpenAI Step 1) Logger setup (From OpenAI implementation) """
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    """ OpenAI Step 2) Random seed setting (From OpenAI implementation) """
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    """ OpenAI Step 3) Environment instantiation """
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    """ OpenAI Step 4) Making placeholders for the computation graph """
    # TODO

    """ OpenAI Step 5) Building the actor-critic computation graph via the actor_critic function passed to
    the algorithm function as an argument """
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    """ OpenAI Step 6) Instantiating the experience buffer """
    # TODO

    """ OpenAI Step 7) Building the computation graph for loss functions and diagnostics specific to the
    algorithm """
    """ OpenAI Step 8) Making training ops """
    """ OpenAI Step 9) Setting up model saving through the logger """
    """ OpenAI Step 10) Defining functions needed for running the main loop of the algorithm (e.g. the core
    update function, get action function, and test agent function, depending on the algorithm) """
    """ OpenAI Step 11) Running the main loop of the algorithm:
        a) Run the agent in the environment
        b) Periodically update the parameters of the agent according to the main equations of
        the algorithm
        c) Log key performance metrics and save agent
    """


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='my_algorithm')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    my_algorithm(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(
            # TODO complete kwargs according to algorithm
        ),
        gamma=args.gamma,
        seed=args.seed,
        steps=args.steps,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
    )
