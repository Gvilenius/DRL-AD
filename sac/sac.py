import numpy as np
import tensorflow as tf
import gym
import time
import core
from core import get_vars
from logx import EpochLogger, restore_tf_graph
from replay_buffer import ReplayBuffer
import mytorcs

"""

Soft Actor-Critic

(With slight variations that bring it closer to TD3)

"""
class SAC(object):
    def __init__(self, env, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.999, 
        polyak=0.995, lr=7e-4, alpha=0.2, batch_size=100, start_steps=10000, 
        max_ep_len=1000, logger_kwargs=dict(), save_freq=1):
        """
        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.

            actor_critic: A function which takes in placeholder symbols 
                for state, ``self.x_ph``, and action, ``self.a_ph``, and returns the main 
                outputs from the agent's Tensorflow computation graph:

                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``mu``       (batch, act_dim)  | Computes mean actions from policy
                                            | given states.
                ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                            | states.
                ``logp_pi``  (batch,)          | Gives log probability, according to
                                            | the policy, of the action sampled by
                                            | ``pi``. Critical: must be differentiable
                                            | with respect to policy parameters all
                                            | the way through action sampling.
                ``q1``       (batch,)          | Gives one estimate of Q* for 
                                            | states in ``self.x_ph`` and actions in
                                            | ``self.a_ph``.
                ``q2``       (batch,)          | Gives another estimate of Q* for 
                                            | states in ``self.x_ph`` and actions in
                                            | ``self.a_ph``.
                ``q1_pi``    (batch,)          | Gives the composition of ``q1`` and 
                                            | ``pi`` for states in ``self.x_ph``: 
                                            | q1(x, pi(x)).
                ``q2_pi``    (batch,)          | Gives the composition of ``q2`` and 
                                            | ``pi`` for states in ``self.x_ph``: 
                                            | q2(x, pi(x)).
                ``v``        (batch,)          | Gives the value estimate for states
                                            | in ``self.x_ph``. 
                ===========  ================  ======================================

            ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
                function you provided to SAC.

            seed (int): Seed for random number generators.

            steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
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

            lr (float): Learning rate (used for both policy and value learning).

            alpha (float): Entropy regularization coefficient. (Equivalent to 
                inverse of reward scale in the original SAC paper.)

            batch_size (int): Minibatch size for SGD.

            start_steps (int): Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.

            max_ep_len (int): Maximum length of trajectory / episode / rollout.

            logger_kwargs (dict): Keyword args for EpochLogger.
        """



        self.logger = EpochLogger(**logger_kwargs)
        # self.logger.save_config(locals())
        self.save_freq = save_freq
        self.lr = lr
        self.t = 0
        self.start_steps = start_steps
        self.batch_size = batch_size
        self.env = env

        tf.set_random_seed(seed)   
        np.random.seed(seed)

        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]

        print(obs_dim)
        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        act_limit = self.env.action_space.high[0]

        # Share information about action space with policy architecture
        ac_kwargs['action_space'] = self.env.action_space

        # Inputs to computation graph
        self.x_ph, self.a_ph, self.x2_ph, self.r_ph, self.d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

        # Main outputs from computation graph
        with tf.variable_scope('main'):
            self.mu, self.pi, logp_pi, q1, q2, q1_pi, q2_pi, v = actor_critic(self.x_ph, self.a_ph, **ac_kwargs)
        
        # Target value network
        with tf.variable_scope('target'):
            _, _, _, _, _, _, _, v_targ = actor_critic(self.x2_ph, self.a_ph, **ac_kwargs)

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

        # Count variables
        var_counts = tuple(core.count_vars(scope) for scope in 
                        ['main/pi', 'main/q1', 'main/q2', 'main/v', 'main'])
        print(('\nNumber of parameters: \t pi: %d, \t' + \
            'q1: %d, \t q2: %d, \t v: %d, \t total: %d\n')%var_counts)

        # Min Double-Q:
        min_q_pi = tf.minimum(q1_pi, q2_pi)

        # Targets for Q and V regression
        q_backup = tf.stop_gradient(self.r_ph + gamma*(1-self.d_ph)*v_targ)
        v_backup = tf.stop_gradient(min_q_pi - alpha * logp_pi)

        # Soft actor-critic losses
        pi_loss = tf.reduce_mean(alpha * logp_pi - q1_pi)
        q1_loss = 0.5 * tf.reduce_mean((q_backup - q1)**2)
        q2_loss = 0.5 * tf.reduce_mean((q_backup - q2)**2)
        v_loss = 0.5 * tf.reduce_mean((v_backup - v)**2)
        value_loss = q1_loss + q2_loss + v_loss
        # Policy train op 
        # (has to be separate from value train op, because q1_pi appears in pi_loss)
        pi_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        pi_params = get_vars('main/pi')
        train_pi_op = pi_optimizer.minimize(pi_loss, var_list=pi_params)
        # Value train op
        # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
        value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        value_params = get_vars('main/q') + get_vars('main/v')
        with tf.control_dependencies([train_pi_op]):
            train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

        # Polyak averaging for target variables
        # (control flow because sess.run otherwise evaluates in nondeterministic order)
        with tf.control_dependencies([train_value_op]):
            target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                    for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        # All ops to call during one training step
        self.step_ops = [pi_loss, q1_loss, q2_loss, v_loss, q1, q2, v, logp_pi, 
                    train_pi_op, train_value_op, target_update]

        # Initializing targets to match main variables
        target_init = tf.group([tf.assign(v_targ, v_main)
                                for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(target_init)
        self.saver = tf.train.Saver(max_to_keep=5)
        # Setup model saving
        # self.logger.setup_tf_saver(self.sess, inputs={'x': self.x_ph, 'a': self.a_ph}, 
        #                             outputs={'mu': self.mu, 'pi': self.pi, 'q1': q1, 'q2': q2, 'v': v})

        self.start_time = time.time()
       

    def _roll_out(self):
        '''
        collect experience in env and update/log each epoch
        '''
        o, r, done, ep_ret, ep_len = self.env.reset(), 0, False, 0, 0
        while not done:
            """
            Until start_steps have elapsed, randomly sample actions
            from a uniform distribution for better exploration. Afterwards, 
            use the learned policy. 
            """
            if self.t >= self.start_steps:
                a = self.get_action(o)
            else:
                a = self.env.action_space.sample()

            # Step the env
            o2, r, done, _ = self.env.step(a)
            ep_ret += r
            ep_len += 1
            # Store experience to replay buffer
            self.replay_buffer.store(o, a, r, o2, done)

            o = o2
            self.t += 1

        return ep_len, ep_ret

    def train(self):
        """
        Perform all SAC updates at the end of the trajectory.
        This is a slight difference from the SAC specified in the
        original paper.
        """
        logger = self.logger
        ep_len, ep_ret = self._roll_out()

        for j in range(ep_len):
            batch = self.replay_buffer.sample_batch(self.batch_size)
            feed_dict = {self.x_ph: batch['obs1'],
                        self.x2_ph: batch['obs2'],
                        self.a_ph: batch['acts'],
                        self.r_ph: batch['rews'],
                        self.d_ph: batch['done'],
                        }
            outs = self.sess.run(self.step_ops, feed_dict)
            logger.store(LossPi=outs[0], LossQ1=outs[1], LossQ2=outs[2],
                        LossV=outs[3], Q1Vals=outs[4], Q2Vals=outs[5],
                        VVals=outs[6], LogPi=outs[7])

        logger.store(EpRet=ep_ret, EpLen=ep_len)
        # if (epoch % save_freq == 0) or (epoch == epochs-1):
        #     logger.save_state({'env': env}, None)
                
    def test(self, epoch):
        logger = self.logger
        # Test the performance of the deterministic version of the agent.
        self.test_agent(1)
        # logger.store(): store the data; logger.log_tabular(): log the data; logger.dump_tabular(): write the data
        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('TestEpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('TestEpLen', average_only=True)
        logger.log_tabular('TotalEnvInteracts', self.t)

        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('Time ', time.time() - self.start_time)
        logger.dump_tabular()


    def get_action(self, o, deterministic=False):
        act_op = self.mu if deterministic else self.pi
        return self.sess.run(act_op, feed_dict={self.x_ph: o.reshape(1,-1)})[0]

    def save(self, fpath, global_step):
        self.saver.save(self.sess, fpath, global_step=global_step)

    def restore(self, fpath):
        self.saver.restore(self.sess, fpath)
        
    def test_agent(self, n=10):
        for j in range(n):
            o, r, d, ep_ret, ep_len = self.env.reset(), 0, False, 0, 0
            while not(d):
                o, r, d, _ = self.env.step(self.get_action(o, True))

                ep_ret += r
                ep_len += 1
            
            self.logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
    def set_target_lane(self, lane):
        self.env.set_target_lane(lane)
            
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MyTorcs-v0')
    parser.add_argument('--hid', type=int, default=300)
    parser.add_argument('--l', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--exp_name', type=str, default='sac_1000_5')
    parser.add_argument('--save_path', type=str, default='model/model')
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--load', type=int, default=1)
    parser.add_argument('--render', type=int, default=0)
    args = parser.parse_args()

    env = gym.make(args.env)
    from run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    start_steps = 10000
    if args.load == 1:
        start_steps = 0
    
    if args.render == 1:
        env.set_test()

    sac =  SAC(env=env , actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs, start_steps=start_steps)



    if args.load == 1:
        # load_path = tf.train.latest_checkpoint("/home/ljf/Desktop/Self-driving/model_no_constrain")
        load_path = tf.train.latest_checkpoint("/home/v/projects/Self-driving/model")
        
        print(load_path)
        sac.restore(load_path)

    if args.train == 0:
        print("test start")
        sac.test_agent(2)
    else:
        print("train start")
        for epoch in range(100000):
            if epoch % 100 == 0:
                import random
                sac.set_target_lane(random.randint(1,5))
    
            sac.train()
            if epoch % 10 == 0:
                sac.test(epoch)
                sac.save(args.save_path, global_step=epoch+1)
