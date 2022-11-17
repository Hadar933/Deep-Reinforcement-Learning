import random
import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
from typing import Tuple, List
from tqdm import tqdm
import numpy as np

class ExperienceReplay:
    def __init__(self, size: int):
        self.size: int = size
        self._exp_rep: deque = deque([], maxlen=size)

    def append(self, experience: Tuple[float, float, float]):
        self._exp_rep.append(experience)

    def sample(self, batch_size: int):
        batched_steps =  random.sample(self._exp_rep, batch_size)
        states = np.stack([b_step[0] for b_step in batched_steps])
        actions = np.array([b_step[1] for b_step in batched_steps])
        rewards = np.array([b_step[2] for b_step in batched_steps])
        next_states = np.stack([b_step[3] for b_step in batched_steps])
        dones = np.array([b_step[4] for b_step in batched_steps])
        return (states, actions, rewards,next_states,dones)


class DQN:
    def __init__(self, exp_rep_size: int, n_episodes: int, n_steps: int, learning_rate: float, hidden_dims: List[int], gamma: float):
        super().__init__()
        self.n_episodes: int = n_episodes
        self.n_steps: int = n_steps
        self.hidden_dims: List[int] = hidden_dims
        self.lr: float = learning_rate
        self.D: ExperienceReplay = ExperienceReplay(exp_rep_size)
        self.env: gym.wrappers.time_limit.TimeLimit = gym.make('CartPole-v1')
        self.Qs_net = self._set_Q_net()
        self.Qs_target = self._set_Q_net()
        self.gamma = gamma

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

    def learn(self, batch_size):
        batch = self.D.sample(batch_size)
        gamma = batch[4]*self.gamma
        y = batch[2]+gamma*np.argmax(self.Qs_target(batch[3]))
        # TODO: Fit() on Qs_net


    def get_action(self, state, epsilon = 0):
        if epsilon > random.random():
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.Qs_net.predict(state))
        return action




if __name__ == '__main__':
    dqn = DQN(exp_rep_size = 10, n_episodes=5, n_steps=10, learning_rate=0.01, hidden_dims = [4,4,4], gamma = 0.99)
    n_episodes = 500
    n_steps = 10
    for ep in tqdm(range(n_episodes)):
        state = dqn.env.reset()
        print(state)
        for step in range(n_steps):
            act = dqn.get_action(np.expand_dims(state,0))
            next_state, reward, done, info = dqn.env.step(act)
            dqn.D.append([state,act,reward,next_state,done])
        dqn.learn(5)
        

            # dqn.learn(batch_size=1, n_ep = 1)
        break
    # s = dqn.env.observation_space.sample()
    # a = dqn.Qs_net.predict(s)