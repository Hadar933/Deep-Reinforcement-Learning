import random
import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
from typing import Tuple, List
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class ExperienceReplay:
    def __init__(self, size: int):
        self.size: int = size
        self._exp_rep: deque = deque([], maxlen=size)

    def append(self, experience: Tuple[float, float, float]):
        self._exp_rep.append(experience)

    def sample(self, batch_size: int):
        rand_sample = random.sample(self._exp_rep, batch_size)
        dict_batch = {
            'states': np.stack([b_step[0] for b_step in rand_sample]),
            'actions': np.array([b_step[2] for b_step in rand_sample]),
            'rewards': np.array([b_step[2] for b_step in rand_sample]),
            'next_states': np.stack([b_step[3] for b_step in rand_sample]),
            'dones': np.array([b_step[4] for b_step in rand_sample])
        }
        return dict_batch

    def __len__(self):
        return self._exp_rep.__len__()


class DQN:
    def __init__(self, exp_rep_size: int, n_episodes: int, n_steps: int, learning_rate: float, hidden_dims: List[int],
                 gamma: float, target_update_interval: int = 10):
        super().__init__()
        self.n_episodes: int = n_episodes
        self.n_steps: int = n_steps
        self.hidden_dims: List[int] = hidden_dims
        self.lr: float = learning_rate
        self.gamma = gamma
        self.D: ExperienceReplay = ExperienceReplay(exp_rep_size)
        self.env: gym.wrappers.time_limit.TimeLimit = gym.make('CartPole-v1')
        self.Qs_net = self._set_Q_net()
        self.Qs_target = self._set_Q_net()
        self.target_update_interval = target_update_interval

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

    def _update_target(self):
        self.Qs_target.set_weights(self.Qs_net.get_weights())

    def learn(self, batch):
        gamma = (1 - batch['dones']) * self.gamma
        y = batch['rewards'] + gamma * np.argmax(self.Qs_target(batch['next_states']))
        y_q = self.Qs_net.predict(batch['states'],verbose = 0)                                      # Predict Qs on all actions
        y_q[np.arange(len(y_q)).tolist(),batch['actions'].astype(int).tolist()]=y       # Change the values of the actual actions to target (y)

        self.Qs_net.fit(batch['states'],y_q, verbose = 0)                                            # loss != 0 only on actual actions takes


    def get_action(self, state, epsilon=0.1):
        if epsilon < random.random():
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.Qs_net.predict(state,verbose= 0))
        return action


if __name__ == '__main__':
    n_episodes = 500
    n_steps = 100
    batch_size = 25
    learning_cycles = 1
    epsilon_decay = 0.99
    print_interval = 10
    min_step_learn = 300
    dqn = DQN(exp_rep_size=1000, n_episodes=n_episodes, n_steps=n_steps, learning_rate=0.1, hidden_dims=[4, 4, 2],
              gamma=0.99, target_update_interval=10)
    epsilon = 1
    ep_reward = 0
    avg_rewards = []
    for ep in tqdm(range(n_episodes)):
        state = dqn.env.reset()
        initialization = len(dqn.D) <= min_step_learn
        for step in range(n_steps):
            action = dqn.get_action(np.expand_dims(state, 0),epsilon = 1 if initialization else epsilon)
            next_state, reward, done, info = dqn.env.step(action)
            dqn.D.append([state, action, reward, next_state, done])
            ep_reward += reward
            if done:
                break
        if ep % print_interval==0:
            avg_reward = ep_reward/dqn.target_update_interval
            print(avg_reward)
            avg_rewards.append(avg_reward)
            ep_reward = 0

        if initialization:
            continue
        for _ in range(learning_cycles):
            batch = dqn.D.sample(batch_size)
            dqn.learn(batch)
        epsilon *= epsilon_decay
        if ep % dqn.target_update_interval==0:
            dqn._update_target()
    plt.plot(avg_rewards)
    plt.show()
