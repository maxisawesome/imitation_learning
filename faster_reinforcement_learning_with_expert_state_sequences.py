import argparse
import sys
import random
import logging
import sys
import gym
import numpy as np
import configparser
from itertools import count
from collections import namedtuple
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from action_value_estimator import ActionValueEstimatorNet
from subgoal_extractor import SubgoalExtractorNet
import utils.util as util

# Set up logging information.
logging.basicConfig(filename='logs/doom.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')
# if gpu is to be used, sets use_cuda to True
use_cuda = torch.cuda.is_available()  #cuda is nvidia's package that moves the computation from the CPU to the GPU
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor #only works with nvidia GPUs!
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
Transition = namedtuple('Transition',
                        ('state', 'subgoal', 'action', 'next_state', 'next_subgoal', 'reward'))

def get_n_step_return(rewards, guidance_rewards, gamma, beta):
    assert (type(rewards) == list)
    assert (type(guidance_rewards) == list)
    assert (type(gamma) == float)
    assert (type(beta) == float)
    if use_cuda:
        guidance_rewards = [guidance_reward.cpu().data.numpy()[0] for guidance_reward in guidance_rewards]
    else:
        guidance_rewards = [guidance_reward.data.numpy()[0] for guidance_reward in guidance_rewards]
    Rn = 0
    for reward, guidance_reward in list(zip(rewards, guidance_rewards))[::-1]:
        Rn = reward + beta * guidance_reward + gamma * Rn 
    Rn = float(Rn)
    return Rn

def epsilon_greedy(q, epsilon):
    assert (type(q).__name__ == 'ndarray')
    assert (type(epsilon) == float)
    p = []
    for i_q in range(len(q)):
        if i_q == np.argmax(q):
            p.append(1 - epsilon + epsilon/ len(q))
        else:
            p.append(epsilon / len(q))
    actions = range(len(q))
    action = np.random.choice(a=actions, p=p)
    return action

def select_action(last_state, state, subgoal, epsilon, subgoal_extractor, action_value_estimator):
    assert (type(state).__name__ == 'ndarray')
    assert (type(subgoal).__name__ == 'Variable')
    assert (type(epsilon) == float)
    assert (type(subgoal_extractor).__name__ == 'SubgoalExtractorNet')
    assert (type(action_value_estimator).__name__ == 'ActionValueEstimatorNet')

    state = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0)
    last_state = torch.from_numpy(last_state).float().unsqueeze(0).unsqueeze(0)
    if use_cuda:
        state = state.cuda()
        last_state = last_state.cuda()
    
    subgoal, guidance_reward = subgoal_extractor((Variable(last_state, volatile=True), Variable(state, volatile=True), subgoal))
    q = action_value_estimator((Variable(state), subgoal))
    action = epsilon_greedy(q.cpu().data.numpy().flatten(), epsilon)
    
    action = action.data[0]
    return action, subgoal, guidance_reward

def optimize_model(memory, action_value_estimator, target_action_value_estimator, subgoal_extractor, ave_optimizer, ave_criterion, batch_size, gamma_n):
    assert (type(memory).__name__ == 'ReplayMemory')
    assert (type(action_value_estimator).__name__ == 'ActionValueEstimatorNet')
    assert (type(target_action_value_estimator).__name__ == 'ActionValueEstimatorNet')
    assert (type(subgoal_extractor).__name__ == 'SubgoalExtractorNet')
    assert (type(ave_optimizer).__name__ == 'Adam')
    assert (type(ave_criterion).__name__ == 'MSELoss')
    assert (type(batch_size) == int)
    assert (type(gamma_n) == float)

    ave_optimizer.zero_grad()
    if len(memory) < batch_size:
        return False
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
    non_final_next_states = non_final_next_states.resize(batch_size, 1, 80, 80)
    state_batch = Variable(torch.cat(batch.state))
    subgoal_batch = Variable(torch.cat(batch.subgoal))
    state_batch = state_batch.resize(batch_size, 1, 80, 80)
    
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))
    next_subgoal_batch = Variable(torch.cat(batch.next_subgoal))

 
    state_action_values = action_value_estimator((state_batch, subgoal_batch)).gather(1, action_batch)


    next_state_values = Variable(torch.zeros(batch_size).type(Tensor))
    next_state_values[non_final_mask] = target_action_value_estimator((non_final_next_states, next_subgoal_batch)).max(1)[0]

    next_state_values.volatile = False
    expected_state_action_values = (next_state_values * gamma_n) + reward_batch

    loss = ave_criterion(state_action_values, expected_state_action_values)

    ave_optimizer.zero_grad()

    loss.backward()

    ave_optimizer.step()
     
    return True
    
def finish_episode(gamma, subgoal_extractor, sge_optimizer, episode_states, episode_rewards):
    assert (type(gamma) == float)
    assert (type(subgoal_extractor).__name__ == 'SubgoalExtractorNet')
    assert (type(sge_optimizer).__name__ == 'Adam')
    assert (type(episode_states) == list)
    assert (type(episode_rewards) == list)

    sge_optimizer.zero_grad()
    R = 0
    transition_policy_loss = []
    discounted_returns = []
    for r in episode_rewards[::-1]: #accumulate the discounted returns 
        R = r + gamma * R
        discounted_returns.insert(0, R)
    discounted_returns = Tensor(discounted_returns)
    subgoal = Variable(torch.zeros([1,1,8]))
    if use_cuda:
        subgoal = subgoal.cuda()
    new_episode = True
    for i in range(len(episode_states)):
        state = torch.from_numpy(episode_states[i]).float().unsqueeze(0).unsqueeze(0)
        if new_episode:
            new_episode = False
            last_state = state
        if use_cuda:
            state = state.cuda()
            last_state = last_state.cuda()
        subgoal, guidance_reward = subgoal_extractor((Variable(last_state), Variable(state), subgoal))
        transition_policy_loss.append(-guidance_reward * discounted_returns[i])
        last_state = state
    sge_optimizer.zero_grad()
    transition_policy_loss = torch.cat(transition_policy_loss).sum()
    transition_policy_loss.backward()
    for param in subgoal_extractor.w_k.parameters():
        param.grad.data.clamp_(-1, 1)
    for param in subgoal_extractor.w_v.parameters():
        param.grad.data.clamp_(-1, 1)
    sge_optimizer.step()

def main():
    parser = argparse.ArgumentParser(description='ICLR 2018 Workshop: "Reproducibility of Faster Reinforcement Learning With Expert State Sequences"')
    parser.add_argument('--config', type=str, metavar='C', required=True,
                        help='File path to configuration file specifying test environment and hyperparameters')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--expert-experiences', type=str, metavar='E', required=True,
                        help='expert experience file')
    parser.add_argument('--savefig', type=str, help='File path for learning curve.')
    parser.add_argument('--save-data-file', type=str, help='File path to save plotting data in csv format')
    parser.add_argument('--beta', type=float, metavar='B', default=0.5)
    parser.add_argument('--gamma', type=float, metavar='G', default=0.99)
    args = parser.parse_args()

    
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
    beta = args.beta


    action_value_estimator = ActionValueEstimatorNet(n_actions=n_actions)
    target_action_value_estimator = ActionValueEstimatorNet(n_actions=n_actions)
    if use_cuda:
        action_value_estimator.cuda()
        target_action_value_estimator.cuda()
    theta = action_value_estimator.state_dict()
    target_action_value_estimator.load_state_dict(theta)

    phi_state_dict = action_value_estimator.get_phi_state_dict()

    expert_experiences = torch.load(args.expert_experiences)

    if use_cuda:
        for i_exp in range(len(expert_experiences)):
            expert_experiences[i_exp] = (expert_experiences[i_exp][0].cuda(), expert_experiences[i_exp][1].cuda())

    subgoal_extractor = SubgoalExtractorNet(expert_experiences=expert_experiences)

    if use_cuda:
        subgoal_extractor = subgoal_extractor.cuda()
  
    subgoal_extractor.update_phi(phi_state_dict)

    ave_optimizer = optim.Adam(action_value_estimator.parameters(), lr=config.getfloat('learning', 'lr'))
    sge_optimizer = optim.Adam([{'params': subgoal_extractor.w_k.parameters()},
                               {'params': subgoal_extractor.w_v.parameters()}],
                               lr=config.getfloat('learning', 'lr'))

    ave_criterion = nn.MSELoss()

    batches_per_epoch = config.getint('learning', 'batches_per_epoch')
    max_epochs = config.getint('defaults', 'max_epochs')

    memory = util.ReplayMemory(config.getint('exp_buffer', 'size'))
    
    epoch = 0
    minibatch_updates = 0
    sync_frequency = config.getint('defaults', 'sync_frequency')
    start_epsilon = config.getfloat('defaults', 'epsilon')
    epsilon_decay = config.getfloat('defaults', 'epsilon_decay')
    epsilon_minimum = config.getfloat('defaults', 'epsilon_minimum')
    n_steps = config.getint('defaults', 'n_steps')
    batch_size = config.getint('learning', 'batch_size')
    sma_reward = None
    sma_rewards = []
    epsilons = []
    steps = 0
    for i_episode in count(1):
        episode_states = []
        episode_subgoals = []
        episode_actions = []
        episode_rewards = []
        episode_guidance_rewards = []
        state = env.reset()
        last_state = state
        subgoal = Variable(torch.zeros(1,1,8), requires_grad=False, volatile=True)
        if use_cuda:
            subgoal = subgoal.cuda()
        for t in count(1):
            steps += 1
            epsilon = max(start_epsilon * (epsilon_decay ** epoch), epsilon_minimum)
            action, subgoal, guidance_reward = select_action(last_state, state, subgoal, epsilon, subgoal_extractor, action_value_estimator) 
            #this gives you the next action and subgoal
            next_state, reward, done, _ = env.step(action) 
            #execute the action just recieved and then update the reward + get next state
            episode_states.append(state)  #store all the important info in lists
            episode_subgoals.append(subgoal)
            episode_actions.append(action)
            episode_rewards.append(reward)
            episode_guidance_rewards.append(guidance_reward)
            n_step_return = get_n_step_return(episode_rewards[-n_steps:], episode_guidance_rewards[-n_steps:], gamma, beta)
            #get the n_step_return for training the action value estimato
            state = next_state 
            if t > n_steps+1: #unkown what htis does
                memory.push(Tensor(episode_states[-n_steps]), episode_subgoals[-n_steps].data, LongTensor([[action]]), Tensor(next_state), subgoal.data, Tensor([n_step_return]))
            if optimize_model(memory, action_value_estimator, target_action_value_estimator, subgoal_extractor, ave_optimizer, ave_criterion, batch_size, gamma_n=gamma**n_steps):
                minibatch_updates += 1
                #Okay bcus optimize_model return a boolean. This should optimize the model for one step.
            if steps % sync_frequency == 0:
                print('Syncing with target network')
                target_action_value_estimator.load_state_dict(action_value_estimator.state_dict())
                print('Syncing subgoal extractor phi with action value estimator')
                phi_state_dict = action_value_estimator.get_phi_state_dict()
                subgoal_extractor.update_phi(phi_state_dict)
            if minibatch_updates > 0 and minibatch_updates % batches_per_epoch == 0:
                epoch += 1
                print('Epoch {} complete. Current Epsilon: {}. Simple Moving Average: {}'.format(epoch, epsilon, sma_reward))
                epsilons.append(epsilon)
                sma_rewards.append(sma_reward)
                #We've reached the end of an epch. Report it to the logger, store episolons + sma_rewards and continue
            if done or epoch > max_epochs:
                if sma_reward is not None:
                    sma_reward = sma_reward * gamma + sum(episode_rewards) * (1 - gamma)
                else:
                    sma_reward = sum(episode_rewards)
                break
            #Done has been marked true or we've reached the maximum number of epochs, record data and break from the loop
            #we're done with the 't loop' (its an episode) now, and this is the only break out of it
            #I wasn't clear on the difference between epoch and episode, so here's a stackexchange answer about it:
            #one episode = one sequence of states, actions and rewards, which ends with terminal state. 
            #For example, playing an entire game can be considered as one episode, the terminal state being 
            #reached when one player loses/wins/draws
            #one epoch = one forward pass and one backward pass of all the training 
            #examples, in the neural network terminology. 

        finish_episode(gamma, subgoal_extractor, sge_optimizer, episode_states, episode_rewards)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}'.format(
                i_episode, t))
            logging.debug('Episode {}\tLast length: {:5d}'.format(
                i_episode, t))
        if epoch >= max_epochs:
            print('Epoch {} complete. Training completed.'.format(epoch))
            if args.savefig:
                plt.plot(list(range(epoch)), sma_rewards, 'o-')
                plt.savefig(args.savefig)
            logging.debug('Epoch {} complete. Training completed.'.format(epoch))
            if args.save_data_file:
                with open(args.save_data_file, 'w') as fd:
                    for i_epoch in range(epoch):
                        fd.write('{},{},{}\n'.format(i_epoch,sma_rewards[i_epoch],epsilons[i_epoch]))
            break

if __name__ == "__main__":
    main()
