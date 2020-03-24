import numpy as np
import torch

from itertools import count

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


def run_perturb(agent,env, method, relative=False, step=0.005, step_cnt=20):
    agent.load()
    res = dict()
    rewards = []
    ep_r = 0
    for i in range(step_cnt):
        perturbation = i*step
        state = env.reset()
        for t in count():
            action = agent.perturb_action(state, perturbation, method=method, relative=relative)
            next_state, reward, done, info = env.step(action)
            ep_r += reward
            if done:
                print("ep_r is {} with epsilon {}".format(ep_r, perturbation))
                rewards.append([perturbation, ep_r])
                ep_r = 0
                break
            state = next_state
    return rewards

def run_test(agent, env, render=False):
    ep_r, t = 0, 0
    if render: env.render()
    state = env.reset()
    gamma = 1
    for t in count():
        action = agent.act(state)
        next_state, reward, done, info = env.step(np.float32(action))
        ep_r += reward * gamma
        gamma *= 0.999
        t += 1
        if done or t == 5000:
            return ep_r, t
        state = next_state
    return 0, 0