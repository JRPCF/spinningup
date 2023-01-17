"""My VPG implementation core.py.

Modified from my_template.
"""

import numpy as np
import scipy.signal

import torch
from torch import nn

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


# TODO: Complete any v functions, q functions, or policies used by MLPActorCritic.

class MLPActorCritic(nn.Module):
    """
    Standard algorithm compatible actor critic.
    """

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64,64), activation=nn.ReLU):
        """Initialize the actor critic.

        Given the information about the environment, initialize policy (self.pi),
        value function (self.pi), and/or any q functions (self.q, self.q1, self.q2).

        Args:
            observation_space
            action_space
            hidden_sizes
            activation
        """
        super().__init__()
        # TODO Implement Function.

    def step(self, obs):
        """ Take a step in space given an observations.

        This function is optional for several algorithms and can be removed.

        Args:
            obs: a numpy array of obs dimension representing an observation.

        Returns:
            A tuple of index length 3.
            At index 0, a numpy array of action_space dimension representing the
            selected action.
            At index 1, a numpy array representing the value of the state.
            At index 2, a numpy array representing the log probability of an
            action.
        """
        # TODO Implement Function.

    def act(self, obs):
        """ Returns an action when given an observation.

        Args:
            obs: a numpy array of obs dimension representing an observation.

        Returns:
            An numpy array of action_space dimension representing the selected
            action.
        """
        # TODO Implement Function.
