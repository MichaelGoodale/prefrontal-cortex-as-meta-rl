import torch
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np
from sklearn.decomposition import PCA

from meta_rl.models import PrefrontalLSTM
from meta_rl.training import train, evaluate
from meta_rl.tasks import TaskOne

import matplotlib.pyplot as plt

def cumulative_regret(probabilities, rewards):
    regrets = [] 
    max_p = max(probabilities)
    reward_sum = 0 
    expected_sum = 0
    for i, r in enumerate(rewards):
        reward_sum += r
        expected_sum += max_p
        regrets.append(expected_sum - reward_sum)
    return regrets

def run_episode(env, model, probs=None, return_values=False, return_infos=False):
    state = env.reset(probs)
    done = False
    reward = 0.0
    rewards = []
    actions = []
    values = []
    infos = []
    action = torch.tensor(0)
    hidden = None
    cell = None
    with torch.no_grad():
        while not done:
            value, action_space, (hidden, cell) = model(state, reward, action, hidden=hidden, cell=cell)
            action_distribution = distributions.Categorical(action_space)
            action = action_distribution.sample()
            state, reward, done, info = env.step(action.item())
            actions.append(action.item())
            rewards.append(reward)
            values.append(value.item())
            infos.append(info)
    return_ = [actions, rewards]
    if return_values:
        return_.append(values)
    if return_infos:
        return_.append(infos)
    return return_

def get_regrets(model_path, probabilities=[0.75, 0.25], n_samples=500):
    model = PrefrontalLSTM(0, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    env = TaskOne(mode='bandit')

    all_regrets = []
    all_actions = []
    all_rewards = []

    for _ in range(500):
        actions, rewards = run_episode(env, model, probs=probabilities.copy())
        all_regrets.append(cumulative_regret(probabilities, rewards))
        all_rewards.append(rewards)
        all_actions.append(actions)

    all_regrets = np.array(all_regrets)
    all_actions = np.array(all_actions)
    all_rewards = np.array(all_rewards)
    return all_regrets, all_actions, all_rewards


def cumulative_regret_plot(independent_bandit_regret, correlated_bandit_regret):
    fig, ax = plt.subplots()
    ax.plot(independent_bandit_regret.mean(axis=0), label="Independent bandits")
    ax.plot(correlated_bandit_regret.mean(axis=0), label="Correlated bandits")
    ax.legend()
    ax.set_xlabel("Trials")
    ax.set_ylabel("Cumulative regret")
    plt.show()

def pca_plot(model_path, probabilities=[0.99, 0.01]):
    model = PrefrontalLSTM(0, 2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    activations = {}
    layer2name = {}
    hook_function = lambda m, i, o: activations[layer2name[m]].append(o)
    for name, layer in model._modules.items():
        layer2name[layer] = name
        activations[name] = []
        layer.register_forward_hook(hook_function)

    env = TaskOne(mode='bandit')

    #because of the seed, we don't need to worry about different PCA projections
    for _ in range(10):
        run_episode(env, model)

    activation_mat = torch.cat([x[0] for x in activations['lstm']]).squeeze().numpy()
    pca = PCA(n_components=2)
    pca.fit(activation_mat)

    activations['lstm'] = []
    actions, _ = run_episode(env, model, probabilities)
    actions = np.array(actions)

    activation_mat = torch.cat([x[0] for x in activations['lstm']]).squeeze().numpy()
    activation_x_y = pca.transform(activation_mat)

    fig, ax = plt.subplots()
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    l_idx = actions == 0
    ax.scatter(activation_x_y[l_idx, 0], activation_x_y[l_idx, 1], c=np.arange(100)[l_idx], cmap='copper')
    ax.scatter(activation_x_y[~l_idx, 0], activation_x_y[~l_idx, 1], c=np.arange(100)[~l_idx], marker='x', cmap='copper')
    ax.set_title(f"P_l={probabilities[0]}, P_r={probabilities[1]}")
    plt.show()
