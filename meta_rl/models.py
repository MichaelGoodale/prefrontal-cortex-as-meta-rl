#Implementation of any models to train the agent with
import gym
import torch

from torch import nn
import torch.nn.functional as F

class PrefrontalLSTM(nn.Module):
    def __init__(self, observation_size, n_actions, hidden_size=48):
        super().__init__()
        self.n_actions = n_actions
        self.lstm = nn.LSTM(input_size=(observation_size+n_actions+1),
                hidden_size=hidden_size)
        self.fc_policy = nn.Linear(hidden_size, n_actions)
        self.fc_value = nn.Linear(hidden_size, 1)

    def forward(self, observation, prev_reward, prev_action, hidden=None, cell=None):
        '''Takes
            observation: Tensor,
            prev_reward: float,
            prev_action: int,

        Outputs:
            value: float,
            action_prob: Tensor[action_number]
        '''
        observation = torch.FloatTensor(observation)
        prev_action = F.one_hot(prev_action.view(-1), self.n_actions).view(-1)
        prev_reward = torch.FloatTensor([prev_reward])
        src = torch.cat((observation, prev_reward, prev_action)).view(1, 1, -1)
        output, (hidden, cells) = self.lstm(src)
        return self.fc_value(output), self.fc_policy(output), (hidden, cells)
