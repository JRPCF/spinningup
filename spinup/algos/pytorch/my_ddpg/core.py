"""My DDPG implementation core.py.

Modified from my_template.
"""

import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

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


class MLPActor(nn.Module):
    """ MLP Actor implementation (only for continuous action spaces because of DDPG).
    
    The actor NN is responsible for learning the policy.
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        """Initialize the categorical actor.

        Given the information about the environment, initialize action limit instance
        variable and an NN with input dimension obs_dim and output dimension act_dim.

        Args:
            obs_dim
            hidden_sizes
            activation
            act_limit
        """
        super().__init__()
        # Tanh improves performance because it doesn't lose information with negative values.
        self.nn = mlp([obs_dim]+list(hidden_sizes)+[act_dim], activation, nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs, act = None):
        """Perform the forward pass of the NN.

        The forward pass performs inference on an observation and scales it according to the
        action limits.

        Args:
            obs
        
        Returns:
            A 1D tensor representing the policy action given the observation.
        """
        return self.act_limit * self.nn(obs)


class MLPQ(nn.Module):
    """ MLP Q function implementation.
    
    The MLP is responsible for estimating the Q function.
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        """Initialize the critic.

        Given the information about the environment and action space,
        initialize an NN estimating the q function with input
        dimension obs_dim+act_dim and output dimension 1.

        Args:
            obs_dim
            act_dim
            hidden_sizes
            activation
        """
        super().__init__()
        self.nn = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        """Perform the forward pass of the NN.

        The forward pass for the NN retrieves the q value for an observation action pair.

        Args:
            obs
            act
        
        Returns:
            A 1D tensor representing the predicted value.
        """
        return torch.squeeze(self.nn(torch.cat([obs, act],dim=-1)), -1) # Critical to ensure v has right shape.

class MLPActorCritic(nn.Module):
    """
    My DDPG actor critic.
    """

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(256,256), activation=nn.ReLU):
        """Initialize the actor critic.

        Given the information about the environment, initialize policy (self.pi)
        and q function (self.q).

        Args:
            observation_space
            action_space
            hidden_sizes
            activation
        """
        super().__init__()

        # For my implementation I will not extend a common actor class to improve readability.
        self.pi = MLPActor(
            observation_space.shape[0],
            action_space.shape[0],
            hidden_sizes,
            activation,
            action_space.high[0],
        )
        self.q  = MLPQ(
            observation_space.shape[0],
            action_space.shape[0],
            hidden_sizes,
            activation
        )

    def act(self, obs):
        """ Returns an action when given an observation.

        This function retrieves the policy distribution from the actor MLP and
        samples an action.

        Args:
            obs: a numpy array of obs dimension representing an observation.

        Returns:
            An numpy array of action_space dimension representing the selected
            action.
        """
        with torch.no_grad(): #disable autograd engine.
            return self.pi(obs).numpy()
