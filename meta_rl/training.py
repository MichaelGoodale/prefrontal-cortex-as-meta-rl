#This contains functions necessary for training.
import gym
import torch

from torch import nn
import torch.nn.functional as F
import torch.distributions as distributions

def get_discounted_returns(rewards, discount_factor):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + R * discount_factor 
        returns.insert(0, R)
    return torch.tensor(returns)

    

def update_model(rewards, values, log_prob_actions, entropies, optimizer, discount_factor=0.9, beta_v=0.05, beta_e=0.05):
    values = torch.cat(values)
    log_prob_actions = torch.cat(log_prob_actions)
    entropies = torch.cat(entropies).squeeze()

    returns = get_discounted_returns(rewards, discount_factor)
    returns = returns.detach()

    td_error = (returns - values)
    policy_loss = log_prob_actions*td_error
    value_loss = td_error*values

    optimizer.zero_grad()
    loss = policy_loss + beta_v * value_loss + beta_e * entropies
    loss = - loss.sum()
    loss.backward()
    optimizer.step()
    return loss.item()

def train(env, model, optimizer, discount_factor=0.9, render=False):
    state = env.reset()
    done = False
    log_prob_actions = []
    values = []
    rewards = []
    entropies = []
    reward = 0.0
    action = torch.tensor(0)

    while not done:
        value, action_space, (hidden, cells) = model(state, reward, action)

        action_probabilities = F.softmax(action_space, dim=-1)
        action_distribution = distributions.Categorical(action_probabilities)
        action = action_distribution.sample()

        log_prob_action = action_distribution.log_prob(action)
        if render:
            env.render()
        state, reward, done, _ = env.step(action.item())
        values.append(value.view(-1))
        rewards.append(reward)
        entropies.append(action_distribution.entropy())
        log_prob_actions.append(log_prob_action.squeeze(0))
    return update_model(rewards, values, log_prob_actions, entropies, optimizer), len(rewards)
