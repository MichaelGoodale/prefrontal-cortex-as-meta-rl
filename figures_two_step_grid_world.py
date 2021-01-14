from cumulative_regret import run_episode
from meta_rl.models import PrefrontalLSTM
from meta_rl.tasks import TwoStep, HumanTwoStep, TwoStepsGridWorld

import numpy as np
import torch
import torch.distributions as distributions
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from tqdm import tqdm
from scipy.stats import spearmanr

def figure_5_b(model_path, N=3, episodes=100):
    model = PrefrontalLSTM(126, 4, hidden_size=192)
    model.load_state_dict(torch.load(model_path))
    commons = []
    uncommons = []
    env = TwoStepsGridWorld(seed=42)
    for seed in range(N):
        n =    {1: {"common": 0, "uncommon": 0},
                0: {"common": 0, "uncommon": 0}}
        stay = {1: {"common": 0, "uncommon": 0},
                0: {"common": 0, "uncommon": 0}} 
        for _ in range(episodes):
            actions, rewards, infos = run_episode(env, model, return_infos=True)
            last_first_position = None
            for info in infos:
                if info != {}:
                    n[info["reward"]][info["state_transition"]] += 1
                    if last_first_position is not None and last_first_position == info["first_position"]:
                        stay[info["reward"]][info["state_transition"]] += 1
                    last_first_position = info["first_position"]

            prev_action = -1

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


import time
def render_episode(model_path):
    model = PrefrontalLSTM(126, 4, hidden_size=192)
    model.load_state_dict(torch.load(model_path))
    commons = []
    uncommons = []
    env = TwoStepsGridWorld(seed=42)
    state = env.reset()
    env.env.render()
    time.sleep(20)
    done = False
    reward = 0.0
    action = torch.tensor(0)
    hidden = None
    cell = None
    with torch.no_grad():
        while not done:
            time.sleep(0.10)
            value, action_space, (hidden, cell) = model(state, reward, action, hidden=hidden, cell=cell)
            action_distribution = distributions.Categorical(action_space)
            action = action_distribution.sample()
            state, reward, done, info = env.step(action.item())
            env.env.render()
render_episode('grid_world_2.pt')
#figure_5_b('grid_world_2.pt', N=2, episodes=20)
