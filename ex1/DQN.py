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

    def learn(self, batch_size):
        pass

    def get_action(self, state, epsilon = 0):
        if epsilon > random.random():
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.Qs_net.predict(state))
        return action




if __name__ == '__main__':
    dqn = DQN(1, 1, 5, 0.01, [1, 2, 3])
    n_episodes = 500
    for ep in tqdm(range(n_episodes)):
        obs = dqn.env.reset()
        for step in range(n_steps):
            act = dqn.get_action(s)
            next_state, reward, done, info = dqn.env.step(act)
            dqn.D.append(state,act,reward,next_state,done)
            dqn.learn(batch_size=1)
            target = reward if done else reward + gamma * np.max(Q[next_state])
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * target
            state = next_state
            if done:  # either reached goal or fell to a hole
                steps_to_goal = step if state == GOAL_STATE else n_steps
                steps_to_goal_arr_over_100_episodes.append(steps_to_goal)
                break
            if step == n_steps - 1 and not done:  # ran out of steps without reaching goal
                steps_to_goal = n_steps
                steps_to_goal_arr_over_100_episodes.append(steps_to_goal)
        if ep % 100 == 0:
            mean_steps = np.mean(steps_to_goal_arr_over_100_episodes)
            mean_steps_to_goal_arr.append(mean_steps.item())
            steps_to_goal_arr_over_100_episodes = []

        epsilon = linear_decay_eps
        ep_rew_arr.append(ep_rew)
    env.close()
    return Q, ep_rew_arr, mean_steps_to_goal_arr
    s = dqn.env.observation_space.sample()
    a = dqn.Qs_net.predict(s)
