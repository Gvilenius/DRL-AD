import argparse
from itertools import count

import json
import os, sys, random
import numpy as np
import json
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from torch.distributions import Normal
from tensorboardX import SummaryWriter
import mytorcs
'''
Implementation of Deep Deterministic Policy Gradients (DDPG) with pytorch 
riginal paper: https://arxiv.org/abs/1509.02971
Not the author's implementation !
'''

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
# OpenAI gym environment name, # ['BipedalWalker-v2', 'Pendulum-v0'] or any continuous environment
# Note that DDPG is feasible about hyper-parameters.
# You should fine-tuning if you change to another environment.
parser.add_argument("--env_name", default="MyTorcs-v0")
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_iteration', default=10, type=int)

parser.add_argument('--learning_rate', default=3e-4, type=float)
parser.add_argument('--gamma', default=0.999, type=int) # discounted factor
parser.add_argument('--capacity', default=100000, type=int) # replay buffer size
parser.add_argument('--batch_size', default=64, type=int) # mini batch size
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=6, type=int)
# optional parameters

parser.add_argument('--sample_frequency', default=256, type=int)
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
parser.add_argument('--exploration_noise', default=0.2, type=float)
parser.add_argument('--max_episode', default=100000, type=int) # num of games
parser.add_argument('--max_length_of_trajectory', default=8000, type=int) # num of games
parser.add_argument('--print_log', default=5, type=int)
parser.add_argument('--update_iteration', default=10, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)
env = gym.make(args.env_name).unwrapped

if args.seed:
    env.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
min_Val = torch.tensor(1e-7).float().to(device) # min value

directory = './exp' + script_name + args.env_name +'./'

class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)


    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 400)
        self.l2 = nn.Linear(400 , 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


def run_test(agent, env, render=False):
    ep_r, t = 0, 0
    # if render: env.render()
    state = env.reset()
    gamma = 1
    for t in count():
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(np.float32(action))
        ep_r += reward * gamma
        gamma *= 0.999
        t += 1
        if done or t == 5000:
            return ep_r, t
        state = next_state
    return 0, 0
class DDPG(object):
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), args.learning_rate)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), args.learning_rate)
        self.replay_buffer = Replay_buffer()
        self.writer = SummaryWriter(directory)
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, s):
        s = torch.FloatTensor(s.reshape(1, -1)).to(device)
        state= Variable(s, requires_grad=True)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def perturb_action(self, s, epsilon = 0.01, method='fgsm', relative=False):
        s = torch.FloatTensor(s.reshape(1, -1)).to(device)
        state= Variable(s, requires_grad=True)
        ori = Variable(s, requires_grad=False)
        delta_s = None

        if method == "random":
            rand = (np.random.randint(0, 2, state.shape)*2 - 1).astype(np.float32)
            if relative:
                delta_s = state.mul(torch.tensor(rand)) * epsilon
            else:
                delta_s = torch.tensor(rand * epsilon)
            state = (state + delta_s).clamp(0, 1)
        elif method == "fgsm":
            Q1 = self.critic(state, self.actor(state))
            Q1.backward()
            g1 = state.grad
            state = Variable(state, requires_grad=True)
            Q2 = self.critic(state, self.actor(state).detach())
            Q2.backward()
            g2 = state.grad
            g = g1 - g2
            if relative:
                delta_s = state.mul(g.sign()) * epsilon
            else:
                delta_s = g.sign() * epsilon 
            state = (state + delta_s).clamp(0, 1)
        else:
            if method == "pgd":
                state = (state + torch.tensor(np.random.uniform(-epsilon, epsilon, state.shape))).type_as(state).clamp(0, 1)
            for i in range(10):
                state = Variable(state, requires_grad=True)
                Q1 = self.critic(state, self.actor(state))
                Q1.backward()
                g1 = state.grad
                state = Variable(state, requires_grad=True)
                Q2 = self.critic(state, self.actor(state).detach())
                Q2.backward()
                g2 = state.grad
                g = g1 - g2
                if relative:
                    delta_s = state.mul(g.sign()) * epsilon
                else:
                    delta_s = g.sign() * epsilon/2
                    delta_s = (state + delta_s - ori).clamp(-epsilon, epsilon)
                state = (state + delta_s).clamp(0, 1)

        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):

        for it in range(args.update_iteration):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(args.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + ((1 - done) * args.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss    
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self):
        torch.save(self.actor.state_dict(), directory + 'actor.pth')
        torch.save(self.critic.state_dict(), directory + 'critic.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")

def run_perturb(agent, method, relative=False, step=0.005, step_cnt=20):
    res = dict()
    rewards = []
    iter_cnt = 1
    if method in ['random', 'pgd']:
        iter_cnt = 10
    for i in range(step_cnt):
        epi_rewards = []
        for _ in range(iter_cnt):
            ep_r = 0
            perturbation = i*step
            state = env.reset()
            for t in count():
                action = agent.perturb_action(state, perturbation, method=method, relative=relative)
                next_state, reward, done, info = env.step(action)
                ep_r += reward
                if done:
                    print("ep_r is {} with epsilon {}".format(ep_r, perturbation))
                    epi_rewards.append(ep_r)
                    ep_r = 0
                    break
                state = next_state
        rewards.append([perturbation, np.mean(epi_rewards)])
                
    return rewards

def main():
    agent = DDPG(state_dim, action_dim)
    ep_r = 0
    if args.mode == 'perturb':
        agent.load()
        res = dict()
        methods = ["pgd", "fgsm"]
        # methods=["pgd"]
        for m in methods:
            res[m] = run_perturb(agent, m, step=0.001, step_cnt=8, relative=False)
        with open("results/DDPG", "w") as f:
            json.dump(res, f)
    elif args.mode == 'test':
        agent.load()
        run_test(agent, env, True)
    elif args.mode == 'train':
        if args.load: agent.load()

        print("====================================")
        print("Collection Experience...")
        while True:
            state = env.reset()
            for t in count():
                action = agent.select_action(state)
                action = (action + np.random.normal(0, args.exploration_noise, size=env.action_space.shape[0])).clip(-1,1)
                next_state, reward, done, info = env.step(action)
                agent.replay_buffer.push((state, next_state, action, reward, np.float(done)))
                state = next_state
                if done or t >= args.max_length_of_trajectory:
                    break
            if len(agent.replay_buffer.storage) >= args.capacity-1:
                break
        print("====================================")
        max_r = -500
        for i in range(args.max_episode):
            state = env.reset()
            for t in count():
                action = agent.select_action(state)
                action = (action + np.random.normal(0, args.exploration_noise, size=env.action_space.shape[0])).clip(-1,1)
                next_state, reward, done, info = env.step(action)
                ep_r += reward
                agent.replay_buffer.push((state, next_state, action, reward, np.float(done)))

                state = next_state
                if done or t >= args.max_length_of_trajectory:
                    agent.writer.add_scalar('ep_r', ep_r, global_step=i)
                    break
            agent.update()

            if i % args.print_log == 0:
                ep_r, t = run_test(agent, env, False) 
                if ep_r > max_r:
                    max_r = ep_r
                    agent.save()
                print("Test Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
            ep_r = 0

    else:
        raise NameError("mode wrong!!!")
    env.close()

if __name__ == '__main__':
    main()