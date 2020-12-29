import torch
import torch.nn.functional as F
import torch.distributions as distributions
import numpy as np
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

def run_episode(env, model, probs=None):
    state = env.reset(probs)
    done = False
    reward = 0.0
    rewards = []
    actions = []
    action = torch.tensor(0)
    hidden = None
    cell = None
    t = 0 
    with torch.no_grad():
        while not done:
            value, action_space, (hidden, cell) = model(state, reward, action, t, hidden=hidden, cell=cell)
            action_distribution = distributions.Categorical(action_space)
            action = action_distribution.sample()
            state, reward, done, _ = env.step(action.item())
            t += 1
            actions.append(action.item())
            rewards.append(reward)
        return actions, rewards

model = PrefrontalLSTM(0, 2)
model.load_state_dict(torch.load('model_bandit_corr.pt'))
model.eval()

env = TaskOne(mode='bandit')
env.set_test()

all_regrets = []
all_actions = []
all_rewards = []

for _ in range(500):
    PROBS=[0.60, 0.40]
    actions, rewards = run_episode(env, model, probs=PROBS)
    all_regrets.append(cumulative_regret(PROBS, rewards))
    all_rewards.append(rewards)
    all_actions.append(actions)

all_regrets = np.array(all_regrets)
all_actions = np.array(all_actions)
all_rewards = np.array(all_rewards)
plt.plot(all_actions.mean(axis=0))
plt.show()
