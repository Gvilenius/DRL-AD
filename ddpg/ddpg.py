import numpy as np
import tensorflow as tf
import gym
import time
import sys
sys.path.append("../common")

class DDPG(object):
    def __init__(self, env, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.999, 
        polyak=0.995, lr=7e-4, alpha=0.2, batch_size=100, start_steps=10000, 
        max_ep_len=1000, logger_kwargs=dict(), save_freq=1):
        pass
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
        pass
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