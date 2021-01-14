from cumulative_regret import run_episode
from meta_rl.models import PrefrontalLSTM
from meta_rl.tasks import TwoStep, HumanTwoStep

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy.stats import spearmanr

def calculate_RPE(rewards, values, gamma=0.9):
    return rewards[-1] + gamma*values[:1] - values[:-1]

def figure_5_b(model_path, N=8, episodes=500):
    model = PrefrontalLSTM(2, 2)
    model.load_state_dict(torch.load(model_path))
    commons = []
    uncommons = []
    for seed in range(N):
        env = TwoStep(seed=seed)
        n =    {1: {"common": 0, "uncommon": 0},
                0: {"common": 0, "uncommon": 0}}
        stay = {1: {"common": 0, "uncommon": 0},
                0: {"common": 0, "uncommon": 0}} 
        for _ in range(episodes):
            actions, rewards, infos = run_episode(env, model, return_infos=True)

            prev_action = -1

            for i, (action, reward, info) in enumerate(zip(actions, rewards, infos)):
                if info['state_transition'] is None:
                    if i >= 2:#Since we can't do it on the first trial
                        n[prev_reward][prev_state_transition] += 1
                        if prev_first_action == first_action:
                            stay[prev_reward][prev_state_transition] += 1

                    prev_first_action = first_action
                    prev_state_transition = state_transition
                    prev_reward = reward 
                else:
                   first_action = action
                   state_transition = info['state_transition']

        commons.append([stay[1]["common"]/n[1]["common"], stay[0]["common"]/n[0]["common"]])
        uncommons.append([stay[1]["uncommon"]/n[1]["uncommon"], stay[0]["uncommon"]/n[0]["uncommon"]])

    commons = np.array(commons)
    uncommons = np.array(uncommons)
    width = 0.35
    gap = 0.05
    plt.bar(np.arange(2) - width / 2 - gap, commons.mean(axis=0),
            width, label='Common', color='blue', zorder=0)
    plt.bar(np.arange(2) + width / 2 + gap, uncommons.mean(axis=0),
            width, label='Uncommon', color='red', zorder=0)
    plt.scatter(np.tile(np.arange(2) - width / 2 - gap, N), commons.reshape(-1), c='black')
    plt.scatter(np.tile(np.arange(2) + width / 2 + gap, N), uncommons.reshape(-1), c='black')
    plt.xticks((0,1), labels=["Rewarded", "Unrewarded"])
    plt.ylim(0.5, 1)
    plt.legend()
    plt.show()

def figure_5_d(model_path, N=8, episodes=500):
    model = PrefrontalLSTM(2, 2)
    model.load_state_dict(torch.load(model_path))
    commons = []
    uncommons = []
    coefs = np.zeros((N, 5, 4))
    for seed in range(N):
        env = TwoStep(seed=seed)
        type_to_idx = {(0, "common"):   [1,0,0,0],
                       (1, "common"):   [0,1,0,0],
                       (0, "uncommon"): [0,0,1,0],
                       (1, "uncommon"): [0,0,0,1]}
        idx_to_type = {0: (0, "common"),
                       1: (1, "common"),
                       2: (0, "uncommon"),
                       3: (1, "uncommon")}
        X = []
        y = []
        for _ in range(episodes):
            actions, rewards, infos = run_episode(env, model, return_infos=True)

            trial_types = []
            first_actions = []
            outcomes = []

            for action, reward, info in zip(actions, rewards, infos):
                if info['state_transition'] is None:
                    if len(trial_types) > 0:
                        outcomes.append(first_actions[-1] == first_actions[-2])
                    trial_types.append(type_to_idx[(reward, state_transition)])
                else:
                    first_actions.append(action)
                    state_transition = info['state_transition']
            for i, outcome in enumerate(outcomes[4:]):
                idx = i+5
                X.append(trial_types[idx-5:idx])
            y += outcomes[4:]
        X = np.array(X) #(N_samples, t, reward_type)
        y = np.array(y)
        l_model = LogisticRegression()
        l_model.fit(X.reshape(-1, 20), y)
        coefs[seed, :, :] = l_model.coef_.reshape(5, 4)

    color = ['red', 'blue']
    linestyle = {"common": '-', "uncommon": ':'}
    for idx, (reward, transition) in idx_to_type.items():
        plt.plot(coefs[:, ::-1, idx].mean(axis=0), c=color[reward], linestyle=linestyle[transition])
        for i in range(N):
            if transition == "common":
                plt.scatter(np.arange(5), coefs[i, ::-1, idx], c=color[reward])
            else:
                plt.scatter(np.arange(5), coefs[i, ::-1, idx], c=color[reward], facecolors='none')
    plt.ylabel('Regression Weights')
    plt.xticks(range(5), labels=range(1, 6))
    plt.xlabel('Trials ago')
    plt.show()

figure_5_b('task_two_20k.pt', N=8, episodes=500)
