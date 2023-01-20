"""My implementation of VPG.

Modified from my_template.
"""

import time
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
import gym
import spinup.algos.pytorch.my_ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

class MyBuffer:
    """
    A standard buffer for storing trajectories experienced by the agent interacting
    with the environment. 
    
    My buffer will use GAE-Lambda (which is used in the Spinning Up
    implementation). This is from the OpenAI implementation.
    """
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        """Initialize buffers and instance variables.

        Args:
            obs_dim
            act_dim
            size
            gamma
            lam
        """
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.p, self.start, self.max = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """Append timestep variables to buffers.

        Args:
            obs
            act
            rew
            logp
        """
        assert self.p < self.max
        self.obs_buf[self.p] = obs
        self.act_buf[self.p] = act
        self.rew_buf[self.p] = rew
        self.val_buf[self.p] = val
        self.logp_buf[self.p] = logp
        self.p += 1

    def finish_path(self, last_val=0):
        """Finish trajectory and record reward and return buffers.

        Args:
            last_val
        """

        path = slice(self.start, self.p)
        r = np.append(self.rew_buf[path], last_val)
        v = np.append(self.val_buf[path], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = r[:-1] + self.gamma * v[1:] - v[:-1]
        self.adv_buf[path] = core.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path] = core.discount_cumsum(r, self.gamma)[:-1]
        self.start = self.p

    def get(self):
        """Retrieve buffers.

        Returns:
            A dictionary including the buffers for observations, actions, calculated returns,
            estimate normalized advantage (calculated from returns), and log probabilities.
        """
        assert self.p == self.max
        self.p, self.path_start_idx = 0, 0
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}


def my_ppo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(),  seed=0, 
        steps=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4, vf_lr=1e-3,
        train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000, target_kl=0.01,
        logger_kwargs=dict(), save_freq=10):
    """
    My Proximal Policy Optimization Implementation

    This is a simple ppo implementation implemented from the SpinningUp pseudocode 
    after reading the original papers with the intention of strengthening my understanding
    of fundatmental RL algorithms.

    This is done from a copy of My VPG implementation and after reading the SpinningUp implementation.

    The function header is the same as the one from the spinningup implementaiton.

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, and a ``pi`` module. The 
            ``step`` method should accept a batch of observations and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to VPG.

        seed (int): Seed for random number generators.

        steps (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

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

    """ OpenAI Step 5) Building the actor-critic computation graph via the actor_critic function passed to
    the algorithm function as an argument """
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Parameter counting and logging from OpenAI implementation.
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    """ OpenAI Step 6) Instantiating the experience buffer """
    # Set up experience buffer
    buf = MyBuffer(obs_dim, act_dim, steps, gamma, lam)

    """ OpenAI Step 7) Building the computation graph for loss functions and diagnostics specific to the
    algorithm """

    def calc_loss_pi(data):
        """Compute the loss function for policy.

        Return the loss and metadata when given a sample from the experience buffer.

        Args:
            data
        
        Returns:
            The loss.
            A dictionary including the approximate KL divergence, the estimate entropy, and
            clipping fraction.
        """
        obs, act, adv, logp_1 = data['obs'], data['act'], data['adv'], data['logp']


        pi, logp_2 = ac.pi(obs, act)
        ratio = torch.exp(logp_2 - logp_1)
        # Note to self: Attempted to reimplement clamp and introduced bugs :(
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio)*adv
        loss_pi =  -(torch.min(ratio * adv, clip_adv)).mean()

        approx_kl = (logp_1 - logp_2).mean().item()
        ent = pi.entropy().mean().item()
        # From OpenAI implementation
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Using nn.MSELoss() doesnt work because of dimensionality
    def calc_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    """ OpenAI Step 8) Making training ops (as suggested: Adam)"""
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    """ OpenAI Step 9) Setting up model saving through the logger """
    # Set up model saving
    logger.setup_pytorch_saver(ac)

    """ OpenAI Step 10) Defining functions needed for running the main loop of the algorithm (e.g. the core
    update function, get action function, and test agent function, depending on the algorithm) """
    
    def update():
        """Perform an update.

        Retrieve an episode, retrieve losses, train the policy NN through gradient ascent (early stopping at
        max KL), train the value function NN and log information.
        """
        data = buf.get()

        # Retrieve inital loss for logging.
        pi_loss_1, pi_info_1 = calc_loss_pi(data)
        v_loss_1 = calc_loss_v(data)

        # Perform Learning on the Policy NN.
        i = 0
        while i < train_pi_iters:
            pi_optimizer.zero_grad()
            pi_loss_2, pi_info_2 = calc_loss_pi(data)
            if pi_info_2['kl'] > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            pi_loss_2.backward()
            pi_optimizer.step()
            i+=1

        logger.store(StopIter=i)

        # Perform Learning on the Value NN.
        i=0
        while i < train_v_iters:
            vf_optimizer.zero_grad()
            v_loss_2 = calc_loss_v(data)
            v_loss_2.backward()
            vf_optimizer.step()
            i+=1


        # Log data.
        kl, ent, cf = pi_info_1['kl'], pi_info_1['ent'], pi_info_2['cf']
        logger.store(
            LossPi=pi_loss_1.item(),
            LossV=v_loss_1.item(),
            KL=kl,
            Entropy=ent,
            ClipFrac=cf,
            DeltaLossPi=(pi_loss_2.item() - pi_loss_1.item()),
            DeltaLossV=(v_loss_2.item() - v_loss_1.item()),
        )


    """ OpenAI Step 11) Running the main loop of the algorithm:"""
    start_time = time.time()
    obs, ep_r, ep_len = env.reset(), 0, 0
    
    for epoch in range(epochs):
        for t in range(steps):
            """ OpenAI Step 11a) Run the agent in the environment """
            # Collect set of trajectories by running policy in environment.
            a, v, logp = ac.step(torch.as_tensor(obs, dtype=torch.float32))
            new_obs, r, done, env_info = env.step(a)

            # Compute rewards-to-go.
            ep_r += r
            ep_len += 1

            # Compute advantage estimate.
            buf.store(obs, a, r, v, logp)
            logger.store(VVals=v)
            obs = new_obs

            timeout = ep_len == max_ep_len
            terminal_step = done or timeout
            ended = terminal_step == steps-1

            """ OpenAI Step 11b) Periodically update the parameters of the agent according to the main equations
            of the algorithm """
            if terminal_step or ended:
                # From OpenAi implementation
                if ended and not(terminal_step):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)

                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or ended:
                    _, v, _ = ac.step(torch.as_tensor(obs, dtype=torch.float32))
                else:
                    v = 0
                # Using total episodic reward as a proxy for value function.
                buf.finish_path(v)
                if terminal_step:
                    logger.store(EpRet=ep_r, EpLen=ep_len)
                # Reset environment.
                obs, ep_r, ep_len = env.reset(), 0, 0

        """ OpenAI Step 11c) Log key performance metrics and save agent copied from SpinningUp implementation."""
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='my_ppo')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    my_ppo(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(
            hidden_sizes=[args.hid]*args.l
        ),
        gamma=args.gamma,
        seed=args.seed,
        steps=args.steps,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
    )
