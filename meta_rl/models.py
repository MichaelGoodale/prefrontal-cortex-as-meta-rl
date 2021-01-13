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
        
        self.initial_hidden = nn.Parameter(torch.zeros(1,1, hidden_size))
        self.initial_cell = nn.Parameter(torch.zeros(1,1, hidden_size))

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
        prev_action = F.one_hot(torch.tensor(prev_action), self.n_actions).view(-1)
        prev_reward = torch.FloatTensor([prev_reward])
        src = torch.cat((observation, prev_reward, prev_action)).view(1, 1, -1)
        if hidden is None and cell is None:
            output, (hidden, cell) = self.lstm(src, (self.initial_hidden, self.initial_cell))
        else:
            output, (hidden, cell) = self.lstm(src, (hidden, cell))
        return self.fc_value(output), F.softmax(self.fc_policy(output), dim=-1), (hidden, cell)
