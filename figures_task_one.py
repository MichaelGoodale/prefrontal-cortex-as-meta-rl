from cumulative_regret import run_episode
from meta_rl.models import PrefrontalLSTM
from meta_rl.tasks import TaskOne

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy.stats import spearmanr

def figure_2_a(model_path):
    model = PrefrontalLSTM(0, 2)
    model.load_state_dict(torch.load(model_path))
    env = TaskOne(mode='monkey')
    env.set_test()

    y = []
    C_r = []
    C_l = []
    R_r = []
    R_l = []
    held_out = []
    for _ in range(500):
        actions, rewards = run_episode(env, model)
        p_l, _ = (env.initial_probability)
        held_out.append((0.1 < p_l < 0.2) or (0.3 < p_l < 0.4))
        C_r.append(sum(actions))
        C_l.append(len(actions) - C_r[-1])
        R_r.append(sum([r for a, r in zip(actions, rewards) if a == 1]))
        R_l.append(sum([r for a, r in zip(actions, rewards) if a == 0]))

    C_r = np.array(C_r)
    C_l = np.array(C_l)
    R_r = np.array(R_r)
    R_l = np.array(R_l)
    held_out = np.array(held_out)

    y = np.log2(C_r / C_l)
    x = np.log2(R_r / R_l)

    plt.scatter(x[held_out], y[held_out], label="Held out parameters", c='red')
    plt.scatter(x[~held_out], y[~held_out], label="Not held out", c='blue')


    lims = [
        np.min([plt.xlim(), plt.ylim()]),  # min of both axes
        np.max([plt.xlim(), plt.ylim()]),  # max of both axes
    ]

    plt.plot(lims, lims, 'k-', zorder=0)
    plt.ylabel(r'$\log_2(\frac{C_R}{C_L})$', fontsize=20)
    plt.xlabel(r'$\log_2(\frac{R_R}{R_L})$', fontsize=20)
    plt.legend()
    plt.show()

def figure_2_b(model_path):
    model = PrefrontalLSTM(0, 2)
    model.load_state_dict(torch.load(model_path))
    env = TaskOne(mode='monkey')
    env.set_test()

    activations = {}
    layer2name = {}
    hook_function = lambda m, i, o: activations[layer2name[m]].append(o)
    for name, layer in model._modules.items():
        layer2name[layer] = name
        activations[name] = []
        layer.register_forward_hook(hook_function)

    action_mat = []
    reward_mat = []
    value_mat = []
    for _ in range(500):
        actions, rewards, values = run_episode(env, model, return_values=True)
        reward_mat.append(rewards)
        action_mat.append(actions)
        value_mat.append(values)
    action_mat = np.array(action_mat)
    value_mat = np.array(value_mat)
    reward_mat = np.array(reward_mat)
    activation_mat = torch.cat([x[0] for i, x in enumerate(activations['lstm']) if i % 100 != 0]).squeeze().numpy()
    value_corr, _ = spearmanr(activation_mat, value_mat[:, 1:].reshape(-1), axis=0)
    action_corr, _ = spearmanr(activation_mat, action_mat[:, :-1].reshape(-1), axis=0)
    reward_corr, _ = spearmanr(activation_mat, reward_mat[:, :-1].reshape(-1), axis=0)
    rewardxaction_corr, _ = spearmanr(activation_mat, reward_mat[:, :-1].reshape(-1) * action_mat[:, :-1].reshape(-1), axis=0)

    value_corr = np.abs(value_corr[:-1, -1])
    action_corr = np.abs(action_corr[:-1, -1])
    reward_corr = np.abs(reward_corr[:-1, -1])
    rewardxaction_corr = np.abs(rewardxaction_corr[:-1, -1])

    plt.bar(range(4), \
            [np.mean(x) for x in [action_corr, reward_corr, rewardxaction_corr, value_corr]], \
            tick_label=[r'$a_{t-1}$', r'$r_{t-1}$',r'$a_{t-1}\times r_{t-1}$', r'$v_t$'], zorder=0)
    plt.ylabel("Correlation")
    plt.scatter([0]*48, action_corr, c='red')
    plt.scatter([1]*48, reward_corr, c='red')
    plt.scatter([2]*48, rewardxaction_corr, c='red')
    plt.scatter([3]*48, value_corr, c='red')
    plt.show()


def figure_2_c(model_path):
    model = PrefrontalLSTM(0, 2)
    model.load_state_dict(torch.load(model_path))
    env = TaskOne(mode='monkey')
    env.set_test()
    X_actions = []
    X_rewards = []
    y = []
    for _ in range(500):
        actions, rewards = run_episode(env, model)
        for i, a in enumerate(actions):
            if i <= 33+15:
                #Only consider data from last 2/3s
                continue
            X_rewards.append(rewards[i-15:i])
            X_actions.append(actions[i-15:i])
            y.append(a)
    X_actions = np.array(X_actions)
    X_rewards = np.array(X_rewards)
    y = np.array(y)

    model = LogisticRegression()
    model.fit(X_actions, y)
    action_coefficients = model.coef_[0]

    model = LogisticRegression()
    model.fit(X_rewards, y)
    reward_coefficients = model.coef_[0]

    fig, axes = plt.subplots(2, 1)
    axes[0].plot(reward_coefficients)
    axes[1].plot(action_coefficients)
    axes[0].set_title("Rewards")
    axes[1].set_title("Actions")
    for ax in axes:
        ax.set_ylabel('Coefficient')
        ax.set_xlabel('Trial lag')
        ax.plot(ax.get_xlim(), (0, 0), linestyle='--', color='black')
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.set_xticks(range(15))
        ax.set_xticklabels(np.arange(15,0, -1))
    plt.show()
figure_2_a('monkey_action_item.pt')
