import random
import torch
import gym
import torch.nn.functional as F
from torch import nn
from collections import deque
from typing import Tuple, List


class ExperienceReplay:
    def __init__(self, size: int):
        self.size: int = size
        self._exp_rep: deque = deque([], maxlen=size)

    def append(self, experience: Tuple[float, float, float]):
        self._exp_rep.append(experience)

    def sample(self, batch_size: int):
        return torch.tensor(random.sample(self._exp_rep, batch_size))


class DQN(nn.Module):
    def __init__(self, exp_rep_size: int, n_episodes: int, n_steps: int, hidden_dims: List[int]):
        super().__init__()
        self.n_episodes: int = n_episodes
        self.n_steps: int = n_steps
        self.hidden_dims: List[int] = hidden_dims
        self.D: ExperienceReplay = ExperienceReplay(exp_rep_size)
        self.env: gym.wrappers.time_limit.TimeLimit = gym.make('CartPole-v1')
        self.Qs_net: nn.ModuleList = self._set_Q_net()

    def _set_Q_net(self) -> nn.ModuleList:
        """
        Our Q_net takes in a state vector (with shape env.observation_space.n) and outputs an action vector
        (with shape env.action_space.n). In can be implemented with arbitrary number of linear hidden layers
        :return: a torch linear model
        """
        Q_net = nn.ModuleList()
        curr_dim = self.hidden_dims[0]
        Q_net.append(nn.Linear(self.env.observation_space.shape[0], curr_dim))
        for next_dim in self.hidden_dims[1:]:
            Q_net.append(nn.Linear(curr_dim, next_dim))
            curr_dim = next_dim
        Q_net.append(nn.Linear(curr_dim, self.env.action_space.n))
        return Q_net

    def forward(self, state_vec):
        for layer in self.Qs_net[:-1]:
            state_vec = F.relu(layer(state_vec))
        action_vec = F.softmax(self.Qs_net[-1](state_vec))
        return action_vec


if __name__ == '__main__':
    dqn = DQN(10, 100, 100, [20, 30, 40, 50])
    s_vec = torch.tensor([1.0, 2.0, 3.0, 4.0])
    a_vec = dqn(s_vec)
    print('hi')
