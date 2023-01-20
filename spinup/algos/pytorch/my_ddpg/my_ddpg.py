"""My implementation of DDPG.

Modified from my_template.
"""

from copy import deepcopy
import time
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import gym
import spinup.algos.pytorch.my_ddpg.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

class MyBuffer:
    """
    A standard buffer for storing trajectories experienced by the agent interacting
    with the environment. 
    
    My buffer will be a simple FIFO buffer as implemented by SpinningUp implementation.

    Here 1 indicates the current time step and 2 indicates the next time step.
    """
    def __init__(self, obs_dim, act_dim, size):
        """Initialize buffers and instance variables.

        Args:
            obs_dim
            act_dim
            size
        """
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.p, self.size, self.max = 0, 0, size

    def store(self, obs, act, rew, obs2, done):
        """Append timestep variables to buffers.
        Overwrite if max exceeded.

        Args:
            obs
            act
            rew
            obs2
            done
        """
        self.obs_buf[self.p] = obs
        self.obs2_buf[self.p] = obs2
        self.act_buf[self.p] = act
        self.rew_buf[self.p] = rew
        self.done_buf[self.p] = done
        self.p = (self.p+1) % self.max
        self.size = min(self.size+1, self.max)

    def sample_batch(self, batch_size=32):
        """Randomly sample a batch of episodes from the buffer.

        From OpenAI implementation

        Args:
            batch_size
        """
        i = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[i],
                obs2=self.obs2_buf[i],
                act=self.act_buf[i],
                rew=self.rew_buf[i],
                done=self.done_buf[i])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


def my_ddpg(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(),  seed=0, 
        steps=4000, epochs=20, replay_size=int(1e6), gamma=0.99, polyak=0.995,pi_lr=1e-3,
        q_lr=1e-3, batch_size = 100, start_steps=10000, update_after=1000, update_every=50,
        act_noise=0.1, num_test_episodes=10, max_ep_len=1000, logger_kwargs=dict(),
        save_freq=1):
    """
    My Deep Deterministic Policy Gradient Implementation

    This is a simple ddpg implementation implemented from the SpinningUp pseudocode 
    after reading the original papers with the intention of strengthening my understanding
    of fundatmental RL algorithms.

    This is done from a copy of My PPO implementation and after reading the SpinningUp implementation.

    The function header is the same as the one from the spinningup implementaiton.

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, and a ``q`` module. The ``act`` method and
            ``pi`` module should accept batches of observations as inputs,
            and ``q`` should accept a batch of observations and a batch of 
            actions as inputs. When called, these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q``        (batch,)          | Tensor containing the current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to DDPG.

        seed (int): Seed for random number generators.

        steps (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    """ OpenAI Step 1) Logger setup (From OpenAI implementation) """
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    """ OpenAI Step 2) Random seed setting (From OpenAI implementation) """
    torch.manual_seed(seed)
    np.random.seed(seed)

    """ OpenAI Step 3) Environment instantiation """
    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    """ OpenAI Step 5) Building the actor-critic computation graph via the actor_critic function passed to
    the algorithm function as an argument """
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)
    # Freeze target network.
    for p in ac_targ.parameters():
        p.requires_grad = False

    """ OpenAI Step 6) Instantiating the experience buffer """
    # Set up experience buffer
    buf = MyBuffer(obs_dim, act_dim, replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    # From OpenAI Implementation
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q])
    logger.log('\nNumber of parameters: \t pi: %d, \t q: %d\n'%var_counts)


    """ OpenAI Step 7) Building the computation graph for loss functions and diagnostics specific to the
    algorithm """

    def calc_loss_pi(data):
        """Compute the loss function for policy.

        Return the loss when given a sample from the replay buffer.

        Args:
            data
        
        Returns:
            The loss.
        """
        o = data['obs']
        q_pi= ac.q(o, ac.pi(o))
        return -q_pi.mean()

    def calc_loss_q(data):
        """Compute the loss function for q function estimator.

        Return the loss and metadata when given a sample from the experience buffer.

        Args:
            data
        
        Returns:
            The loss.
        """
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q = ac.q(o, a)

        #Bellman backup.
        with torch.no_grad():
            q_pi_targ = ac_targ.q(o2, ac_targ.pi(o2))
            backup = r + gamma * (1 - d) * q_pi_targ

        loss_q = ((q-backup)**2).mean()

        loss_info = dict(QVals=q.detach().numpy())

        return loss_q, loss_info


    """ OpenAI Step 8) Making training ops (as suggested: Adam)"""
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(ac.q.parameters(), lr=q_lr)

    """ OpenAI Step 9) Setting up model saving through the logger """
    # Set up model saving
    logger.setup_pytorch_saver(ac)

    """ OpenAI Step 10) Defining functions needed for running the main loop of the algorithm (e.g. the core
    update function, get action function, and test agent function, depending on the algorithm) """
    
    def update(data):
        """Perform an update.

        Given an episode, perform one gradient descent step on Q and one on P (with a frozen set of Q network
        parameter to save computational effort), and update the target network with polyak averaging.

        Args:
        data
        """

        # Run one gradient descent step on Q.
        q_optimizer.zero_grad()
        loss_q, loss_info = calc_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Freeze the Q-network.
        for p in ac.q.parameters():
            p.requires_grad = False

        # Run one gradient descent on Pi.
        pi_optimizer.zero_grad()
        loss_pi = calc_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze the Q-network.
        for p in ac.q.parameters():
            p.requires_grad = True

        logger.store(
            LossQ=loss_q.item(),
            LossPi=loss_pi.item(),
            **loss_info
        )

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # From OpenAI implementation
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, noise_scale):
        """Retrieve an action.

        Given an action, retrieve an  action and add noise and clipping.

        Args:
        obs,
        noise_scale

        Return:
        a numpy array of actions
        """
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)
    
    def test_agent():
        """Run the test environment"""
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    """ OpenAI Step 11) Running the main loop of the algorithm:"""
    total_steps = steps * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        """ OpenAI Step 11a) Run the agent in the environment """
        # Randomly sample sactions from a uniform distribution for better exploration
        # until the start step then use policy.
        if t > start_steps:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()

        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # From OpenAI implementation
        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        buf.store(o, a, r, o2, d)

        o = o2

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        """ OpenAI Step 11b) Periodically update the parameters of the agent according to the main equations
        of the algorithm """

        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = buf.sample_batch(batch_size)
                update(data=batch)

        if (t+1) % steps == 0:
            epoch = (t+1) // steps
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            test_agent()

            """ OpenAI Step 11c) Log key performance metrics and save agent copied from SpinningUp implementation."""
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('QVals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='my_ddpg')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    my_ddpg(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(
            hidden_sizes=[args.hid]*args.l
        ),
        gamma=args.gamma,
        seed=args.seed,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
    )
