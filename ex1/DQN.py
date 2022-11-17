import random
import torch
import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
from typing import Tuple, List

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ExperienceReplay:
    def __init__(self, size: int):
        self.size: int = size
        self._exp_rep: deque = deque([], maxlen=size)

    def append(self, experience: Tuple[float, float, float]):
        self._exp_rep.append(experience)

    def sample(self, batch_size: int):
        return random.sample(self._exp_rep, batch_size)


class DQN:
    def __init__(self, exp_rep_size: int, n_episodes: int, n_steps: int, learning_rate: float, hidden_dims: List[int]):
        super().__init__()
        self.n_episodes: int = n_episodes
        self.n_steps: int = n_steps
        self.hidden_dims: List[int] = hidden_dims
        self.lr: float = learning_rate
        self.D: ExperienceReplay = ExperienceReplay(exp_rep_size)
        self.env: gym.wrappers.time_limit.TimeLimit = gym.make('CartPole-v1')
        self.Qs_net = self._set_Q_net()

    def _set_Q_net(self):
        """
        Our Q_net takes in a state vector (with shape env.observation_space.n) and outputs an action vector
        (with shape env.action_space.n). In can be implemented with arbitrary number of linear hidden layers
        :return: a torch linear model
        """
        Q_net = Sequential()
        Q_net.add(Dense(self.hidden_dims[0], input_dim=self.env.observation_space.shape[0], activation='relu'))
        for next_dim in self.hidden_dims[1:]:
            Q_net.add(Dense(next_dim, activation='relu'))
        Q_net.add(Dense(self.env.action_space.n, activation='softmax'))
        Q_net.compile(loss='mse', optimizer=Adam(learning_rate=self.lr))
        return Q_net


if __name__ == '__main__':
    dqn = DQN(1, 1, 5, 0.01, [1, 2, 3])
    s = dqn.env.observation_space.sample()
    a = dqn.Qs_net.predict(s)
