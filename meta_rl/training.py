#This contains functions necessary for training.
import gym
import torch

from torch import nn
import torch.nn.functional as F
import torch.distributions as distributions

from .utils import plot_grad_flow

def get_discounted_returns(rewards, values, discount_factor):
    returns = []
    advantages = []
    advantage = 0
    R = values[-1].item()
    for i in reversed(range(len(rewards))):
        r = rewards[i]
        v = values[i]
        n_v = values[i+1]
        R = r + R * discount_factor
        deltas = r + n_v * discount_factor - v
        advantage = advantage * discount_factor + deltas
        returns.insert(0, R)
        advantages.insert(0, advantage)
    return torch.tensor(returns), torch.tensor(advantages)

    

def update_model(model, rewards, values, log_prob_actions, entropies, optimizer, discount_factor=0.9, beta_v=0.05, beta_e=0.05):
    values = torch.cat(values)
    log_prob_actions = torch.cat(log_prob_actions)
    entropies = torch.stack(entropies)

    returns, advantages = get_discounted_returns(rewards, values, discount_factor)
    policy_loss = -(log_prob_actions*advantages.detach()).sum()
    value_loss = F.smooth_l1_loss(values[:-1], returns, reduction='sum')
    optimizer.zero_grad()
    loss = policy_loss + beta_v * value_loss - beta_e * entropies.sum()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()

def evaluate(env, model, render=False):
    state = env.reset()
    done = False
    reward = 0.0
    rewards = []
    actions = []
    action = torch.tensor(0)

    with torch.no_grad():
        while not done:
            value, action_space, (hidden, cells) = model(state, reward, action)
            action_probabilities = F.softmax(action_space, dim=-1)
            action_distribution = distributions.Categorical(action_probabilities)
            action = action_distribution.sample()
            if render:
                env.render()
            state, reward, done, _ = env.step(action.item())
            actions.append(action.item())
            rewards.append(reward)
        return rewards, actions

def train(env, model, optimizer, discount_factor=0.9, render=False):
    state = env.reset()
    done = False
    log_prob_actions = []
    values = []
    rewards = []
    entropies = []
    actions = []
    reward = 0.0
    action = torch.tensor(0)
    hidden = None
    cell = None
    t = 0

    while not done:
        if render:
            env.render()

        value, action_space, (hidden, cell) = model(state, reward, action.item(), hidden=hidden, cell=cell)
        action_distribution = distributions.Categorical(action_space)
        action = action_distribution.sample()
        log_prob_action = action_distribution.log_prob(action)
        log_prob_actions.append(log_prob_action.squeeze(0))
        entropies.append(action_distribution.entropy())


        state, reward, done, _ = env.step(action.item())
        values.append(value.view(-1))
        actions.append(action.item())
        rewards.append(reward)
        t += 1
    value, action_space, (hidden, cell) = model(state, reward, action, hidden=hidden, cell=cell)
    values.append(value.view(-1))
    return update_model(model, rewards, values, log_prob_actions, entropies, optimizer, discount_factor), rewards, actions
