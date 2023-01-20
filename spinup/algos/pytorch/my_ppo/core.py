"""My VPG implementation core.py.

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


class MLPCategoricalActor(nn.Module):
    """ MLP Actor implementation for discrete action spaces.
    
    The actor NN is responsible for learning the policy.
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        """Initialize the categorical actor.

        Given the information about the environment, initialize an NN with input dimension 
        obs_dim and output dimension act_dim. The categorical policy behaves as a classifier
        over discrete actions.

        Args:
            obs_dim
            hidden_sizes
            activation
        """
        super().__init__()
        self.nn = mlp([obs_dim]+list(hidden_sizes)+[act_dim], activation)

    def _dist(self, obs):
        """Calculate the categorical distribution given an observation.

        Args:
            obs
        
        Returns:
            A torch.distributions object representing the policy distribution
            given the observation.
        """
        return torch.distributions.Categorical(logits=self.nn(obs))
    
    def _logp(self, pi, a):
        """Wrap log probability for policy.

        Args:
            pi
            a
        
        Returns:
            log_probability of action
        """
        return pi.log_prob(a)

    def forward(self, obs, act = None):
        """Perform the forward pass of the NN.

        The forward pass for the NN retrieves the policy distribution
        given an observation and, if an action is passed, also returns the
        log probability of said action in the given distribution.

        Args:
            obs
            act
        
        Returns:
            A torch.distributions object representing the policy distribution
            given the observation.
            A 1D tensor representing the log probability.
        """
        pi = self._dist(obs)
        logp_a = None
        if act is not None:
            logp = pi.log_prob(act)
        return pi, logp

class MLPGaussianActor(nn.Module):
    """ MLP Actor implementation for continous action spaces.
    
    The actor MLP is responsible for learning the policy.
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        """Initialize the gaussian actor.

        Given the information about the environment, initialize an NN estimating the mean of the distribution
        with input dimension obs_dim and output dimension act_dim and an instance variable for the log standard
        deviation of the distibution. The diagonal gaussian policy NN maps from observations to mean actions.

        Args:
            obs_dim
            hidden_sizes
            activation
        """
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.nn = mlp([obs_dim]+list(hidden_sizes)+[act_dim], activation)
    
    def _dist(self, obs):
        """Calculate the diagonal gaussian distribution given an observation.

        Implementation taken from my solution to exercise 1.1. Gaussian normal
        taken from estimating the mean through the NN and retrieving the standard
        deviation by exponentiating the log standard deviation.

        Args:
            obs
        
        Returns:
            A torch.distributions object representing the policy distribution
            given the observation.
        """
        return torch.distributions.Normal(
            self.nn(obs),
            torch.exp(self.log_std)
        )

    def _logp(self, pi, a):
        """Wrap log probability for policy.

        Args:
            pi
            a
        
        Returns:
            log_probability of action
        """
        return pi.log_prob(a).sum(-1)

    def forward(self, obs, act = None):
        """Perform the forward pass of the NN.

        The forward pass for the NN retrieves the policy distribution
        given an observation and, if an action is passed, also returns the
        log probability of said action in the given distribution.

        Args:
            obs
            act
        
        Returns:
            A torch.distributions object representing the policy distribution
            given the observation.
            A 1D tensor representing the log probability.
        """
        pi = self._dist(obs)
        logp_a = None
        if act is not None:
            logp = pi.log_prob(act).sum(-1)
        return pi, logp

class MLPCritic(nn.Module):
    """ MLP Critic implementation.
    
    The critic MLP is responsible for learning the value function.
    """
    def __init__(self, obs_dim, hidden_sizes, activation):
        """Initialize the critic.

        Given the information about the environment, initialize an NN estimating the value function with input
        dimension obs_dim and output dimension 1.

        Args:
            obs_dim
            hidden_sizes
            activation
        """
        super().__init__()
        self.nn = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        """Perform the forward pass of the NN.

        The forward pass for the NN retrieves the value for an observation.

        Args:
            obs
        
        Returns:
            A 1D tensor representing the predicted value.
        """
        return torch.squeeze(self.nn(obs), -1) # Critical to ensure v has right shape.

class MLPActorCritic(nn.Module):
    """
    My VPG actor critic.
    """

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64,64), activation=nn.ReLU):
        """Initialize the actor critic.

        Given the information about the environment, initialize policy (self.pi)
        and value function (self.v).

        Args:
            observation_space
            action_space
            hidden_sizes
            activation
        """
        super().__init__()

        # For my implementation I will not extend a common actor class to improve readability.
        if isinstance(action_space, Box):
            # Given a continuous action space, initialize policy to gaussian actor.
            self.pi = MLPGaussianActor(
                observation_space.shape[0],
                action_space.shape[0],
                hidden_sizes,
                activation,
            )
        elif isinstance(action_space, Discrete):
            # Given a discrete action space, initialize policy to categorical actor.
            self.pi = MLPCategoricalActor(
                observation_space.shape[0],
                action_space.n,
                hidden_sizes,
                activation,
            )
        self.v  = MLPCritic(
            observation_space.shape[0],
            hidden_sizes,
            activation
        )


    def step(self, obs):
        """ Take a step in space given an observations.

        This function retrieves the policy distribution from the actor MLP, samples
        an action and recovers the log probability of said action (adjusting the 
        dimension as in exercise 1.1).

        Args:
            obs: a numpy array of obs dimension representing an observation.

        Returns:
            A tuple of index length 3.
            At index 0, a numpy array of action_space dimension representing the
            selected action.
            At index 1, a numpy array of dimension 1 representing the value of the
            observation.
            At index 2, a numpy array representing the log probability of an
            action.
        """
        with torch.no_grad(): #disable autograd engine.
            pi = self.pi._dist(obs)
            a = pi.sample()
            logp=self.pi._logp(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp.numpy()

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
            pi = self.pi._dist(obs)
            a = pi.sample()
        return a.numpy()

