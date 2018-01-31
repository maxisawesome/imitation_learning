import gym
import datetime

import configparser
import argparse
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as T
import logging

import utils.util as util

# Define Doom environment and create wrapper.
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete
from scipy.misc import imresize
from gym.core import ObservationWrapper
from gym.spaces.box import Box

from dqn import DQN

# Set up logging information.
logging.basicConfig(filename='logs/doom.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [] 
        self.position = 0
      
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

def epsilon_greedy(q, epsilon):
    p = []
    
    for i_q in range(len(q)):
        if i_q == np.argmax(q):
            p.append(1 - epsilon + epsilon/ len(q))
        else:
            p.append(epsilon / len(q))
    actions = range(len(q))
    action = np.random.choice(a=actions, p=p)
    return action

def select_action(state, epsilon, dqn):
    q = dqn(state)
    if use_cuda:
        action = epsilon_greedy(q.cpu().data.numpy()[0], epsilon)
    else:
        action = epsilon_greedy(q.data.numpy()[0], epsilon)
    action = action.data[0]
    return action

def optimize_model(memory, batch_size, gamma, criterion, optimizer, dqn, target_dqn):
    if len(memory) < 1000: #batch_size:
        return False
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    
    # Compute a mask of non-final states and concatenates the batch elements
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))
    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
    non_final_next_states = non_final_next_states.resize(batch_size, 1, 80, 80)

    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))
    
    # Compute Q(s_t, a) - model computes Q(s_t), then we select the
    # columns of actions taken
    state_batch = state_batch.resize(batch_size, 1, 80, 80)
    state_action_values = dqn(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(batch_size).type(Tensor))
    next_state_values[non_final_mask] = target_dqn(non_final_next_states).max(1)[0]
 #   next_state_values[non_final_mask] = dqn(non_final_next_states).max(1)[0]
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch
    
    loss = criterion(state_action_values, expected_state_action_values)
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    return True

def main():
    parser = argparse.ArgumentParser(description='ICLR 2018 Workshop: "Reproducibility of Faster Reinforcement Learning With Expert State Sequences -- DQN Baseline"')
    parser.add_argument('--mode', type=str, default='train', help='[train|test]')
    parser.add_argument('--config', type=str, metavar='C', required=True,
                        help='File path to configuration file specifying test environment and hyperparameters')
    parser.add_argument('--seed', type=int, metavar='S',
                        help='random seed')
    parser.add_argument('--render', type=bool, default=False, metavar='R', help='Whether the environment will be rendered after each step')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--savefig', type=str, help='File path for learning curve.')
    parser.add_argument('--save-data-file', type=str, help='File path to save plotting data in csv format')
    parser.add_argument('--gamma', type=float, metavar='G', default=0.99)
    parser.add_argument('--save-model', type=str, help='File path prefix to save best model, will append timestamp and .pt')
    parser.add_argument('--load-model', type=str, help='File path to saved model which will be loaded')
    parser.add_argument('--save-exp', type=str, help='File path to save expert experience.')
    args = parser.parse_args()

    if args.mode not in ['train','test']:
        print('Invalid mode: {}'.format(args.mode))
        sys.exit()
    mode = args.mode

    
    config = configparser.RawConfigParser()
    config.read(args.config)

    env = util.make_env(config.get('defaults', 'env'), config.get('defaults', 'env_name'))
    
    if args.seed is not None:
        seed = args.seed
    else:
        seed = random.randint(0,1000)
    print('Using random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)

    n_actions = env.action_space.n
    
    gamma = args.gamma

    dqn = DQN(n_actions=n_actions)
    target_dqn = DQN(n_actions=n_actions)
    if use_cuda:
        dqn.cuda()
        target_dqn.cuda()
    theta = dqn.state_dict()
    target_dqn.load_state_dict(theta)

    if mode == 'test':
        print('Loading model for testing: {}'.format(args.load_model))
        dqn = torch.load(args.load_model)

    optimizer = optim.Adam(dqn.parameters(), lr=config.getfloat('learning', 'lr'))

    criterion = nn.MSELoss()

    memory = ReplayMemory(config.getint('exp_buffer', 'size'))

    batches_per_epoch = config.getint('learning', 'batches_per_epoch')
    max_epochs = config.getint('defaults', 'max_epochs')

    epoch = 0
    minibatch_updates = 0
    sync_frequency = config.getint('defaults', 'sync_frequency')
    start_epsilon = config.getfloat('defaults', 'epsilon')
    epsilon_decay = config.getfloat('defaults', 'epsilon_decay')
    epsilon_minimum = config.getfloat('defaults', 'epsilon_minimum')
    epsilon_test = config.getfloat('defaults', 'epsilon_test')
    n_steps = config.getint('defaults', 'n_steps')
    batch_size = config.getint('learning', 'batch_size')
    #TODO: Implement target network.
    sma_reward = None
    sma_rewards = []
    epsilons = []
    best_sma_reward = None
    steps = 0
    if mode == 'train':
        for i_episode in count(1):
            state = env.reset()
            state = Tensor(state)
            temp = state.clone()
            temp.resize_(1,1,80,80)
            temp.copy_(state)
            state = temp
            #state.resize_(1,1,80,80)

            episode_rewards = []
            for t in count(1):
                steps += 1
                epsilon = max(start_epsilon * (epsilon_decay ** epoch), epsilon_minimum)
                action = select_action(Variable(state), epsilon, dqn)
                next_state, reward, done, _ = env.step(action)
                next_state = Tensor(next_state)
                temp = next_state.clone()
                temp.resize_(1,1,80,80)
                temp.copy_(state)
                next_state = temp
                #next_state.resize_(1,1,80,80)
                episode_rewards.append(reward)
                reward = Tensor([reward]) # NOTE: There may be an issue here with only putting reward in list not 2 lists.
                
                memory.push(state, LongTensor([[action]]), next_state, reward)
    
                state = next_state
            
                if optimize_model(memory, batch_size, gamma, criterion, optimizer, dqn, target_dqn):
                    minibatch_updates += 1
                if steps % sync_frequency == 0:
                    print('Syncing with target network')
                    target_dqn.load_state_dict(dqn.state_dict())
                if minibatch_updates > 0 and minibatch_updates % batches_per_epoch == 0:
                    epoch += 1
                    print('Epoch {} complete. Current Epsilon: {}. Simple Moving Average: {}'.format(epoch, epsilon, sma_reward))
                    epsilons.append(epsilon)
                    sma_rewards.append(sma_reward)
                    if (best_sma_reward is None or sma_reward > best_sma_reward) and args.save_model is not None:
                        best_sma_reward = sma_reward
                        best_model_file = '{}_{}_{}sma.pt'.format(args.save_model, datetime.datetime.now().strftime("%Y-%m-%d-%Hh-%Mm-%Ss"), best_sma_reward)
                        print('Saving model: {}\n'.format(best_model_file))
                        torch.save(dqn, best_model_file)
                if done or epoch >= max_epochs:
                    if sma_reward is not None:
                        sma_reward = sma_reward * gamma + sum(episode_rewards) * (1 - gamma)
                    else:
                        sma_reward = sum(episode_rewards)
                    break
                
            if i_episode % args.log_interval == 0:
                print('Episode {}\tLast length: {:5d}'.format(
                    i_episode, t))
                logging.debug('Episode {}\tLast length: {:5d}'.format(
                    i_episode, t))
            if epoch >= max_epochs:
                print('Max Epoch ({}) complete. Training completed.'.format(epoch))
                if args.savefig:
                    plt.plot(list(range(epoch)), sma_rewards, 'o-')
                    plt.savefig(args.savefig)
                logging.debug('Max Epoch ({}) complete. Training completed.'.format(epoch))
                if args.save_data_file:
                    with open(args.save_data_file, 'w') as fd:
                        for i_epoch in range(epoch):
                            fd.write('{},{},{}\n'.format(i_epoch,sma_rewards[i_epoch],epsilons[i_epoch]))
                break
    elif mode == 'test':
        expert_experiences = []
        max_experiences = 10000
        save_experiences_file = args.save_exp
        for i_episode in range(5000):
            state = env.reset()
            state = Tensor(state)
            temp = state.clone()
            temp.resize_(1,1,80,80)
            temp.copy_(state)
            state = temp
            #state.resize_(1,1,80,80)
            episode_rewards = []
            for t in count(1):
                epsilon = epsilon_test
                action = select_action(Variable(state), epsilon, dqn)
                next_state, reward, done, _ = env.step(action)
                next_state = Tensor(next_state)
                temp = next_state.clone()
                temp.resize_(1,1,80,80)
                temp.copy_(state)
                next_state = temp
                #next_state.resize_(1,1,80,80)
                episode_rewards.append(reward)
                reward = Tensor([reward]) # NOTE: There may be an issue here with only putting reward in list not 2 lists.
                
                expert_experience = (Variable(state), Variable(next_state))
                expert_experiences.append(expert_experience)

                state = next_state
            
                if done or len(expert_experiences) > max_experiences:
                    if sma_reward is not None:
                        sma_reward = sma_reward * gamma + sum(episode_rewards) * (1 - gamma)
                    else:
                        sma_reward = sum(episode_rewards)
                    break
                
            if i_episode % args.log_interval == 0:
                print('Episode {}\tLast length: {:5d}'.format(
                    i_episode, t))
                logging.debug('Episode {}\tLast length: {:5d}'.format(
                    i_episode, t))
                print('Simple Moving Average: {0:.2f}'.format(sma_reward))
                print('Expert Experiences Gathered: {}'.format(len(expert_experiences)))
            if len(expert_experiences) > max_experiences:
                break
        torch.save(expert_experiences, save_experiences_file)
        
    else:
        print('Error: Invalid mode {}, must be one of [train|test]'.format(mode))

if __name__ == "__main__":
    main()
