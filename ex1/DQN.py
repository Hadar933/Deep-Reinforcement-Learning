import random
import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop
from collections import deque
from typing import Tuple, List, Union
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


OPTIMIZERS={
    'Adam': Adam,
    'RMSprop': RMSprop
}


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

class DQN():

    def __init__(
                self, env : gym.Env, hidden_dims: List[int] = [16,32,8], lr : float = 0.001 , epsilon: float = 0.1, gamma : float = 0.95, 
                learning_epochs: int = 1, batch_size : int = 32, target_update_interval: int =2, steps_per_epoch: int = 100, 
                buffer_size : int =10000, min_steps_learn: int = 100, inner_activation: str = 'relu', verbose : Union[str,int] = 0, 
                final_activation : str = 'softmax', optimizer_name: str = 'Adam' , loss_fn_name : str = 'mse', 
                kernel_initializer: str = 'glorot_normal', report_interval = 3):
        assert optimizer_name in OPTIMIZERS.keys() ; "Unknown optimizer"
        self.env = env
        self.action_space   = env.action_space.n
        self.state_space    = env.observation_space.shape[0]
        self.steps_per_epoch = steps_per_epoch
        self.hidden_dims = hidden_dims
        self.target_update_interval = target_update_interval
        self.epsilon = epsilon
        self.gamma = gamma
        self.lr = lr
        self.min_steps_learn = min_steps_learn
        self.inner_act = inner_activation
        self.final_activation = final_activation
        self.optimizer_name = optimizer_name
        self.loss_fn_name= loss_fn_name
        self.kernel_initializer = kernel_initializer
        self.verbose = verbose
        self.learning_epochs = learning_epochs
        self.batch_size = batch_size
        self.report_interval = report_interval
        self.replay_buffer = ExperienceReplay(buffer_size)
        self.q = self._build_model()
        self.q_target = self._build_model()
    
    def _build_model(self):
        net = Sequential()
        net.add(Dense(self.hidden_dims[0], input_dim=self.state_space, activation=self.inner_act))
        for next_dim in self.hidden_dims[1:]:
            net.add(Dense(next_dim, activation=self.inner_act, kernel_initializer=self.kernel_initializer))
        net.add(Dense(self.env.action_space.n, activation=self.final_activation, kernel_initializer=self.kernel_initializer))
        net.compile(loss=self.loss_fn_name, optimizer=OPTIMIZERS[self.optimizer_name]())
        return net
    
    def _update_target(self):
        self.q_target.set_weights(self.q.get_weights())

    def __call__(self, state, epsilon = None):
        epsilon  = self.epsilon if epsilon is None else epsilon
        if epsilon < random.random():
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q.predict(state,verbose= self.verbose))
        return action

    def learn(self):
        batch = self.replay_buffer.sample(self.batch_size)
        gamma = (1 - batch['dones']) * self.gamma
        y = batch['rewards'] + gamma * np.argmax(self.q_target(batch['next_states']))
        y_q = self.q.predict(batch['states'],verbose = 0)                                      # Predict Qs on all actions
        y_q[np.arange(len(y_q)).tolist(),batch['actions'].astype(int).tolist()]=y       # Change the values of the actual actions to target (y)
        loss = self.q.fit(batch['states'],y_q, verbose = 0)                                            # loss != 0 only on actual actions takes
        return loss

    def collect_batch(self, n_steps,epsilon =None, show_progress = False):
        ep_lengths = []
        episodes = 1
        episode_steps = 0
        state = self.env.reset()
        ep_reward=0
        if show_progress:
            pbar = tqdm(total=n_steps)
        for step_num in range(n_steps):
            action = self(np.expand_dims(state, 0),epsilon)
            next_state, reward, done, info = self.env.step(action)
            self.replay_buffer.append([state, action, reward, next_state, done])
            ep_reward += reward
            episode_steps +=1
            if done:
                state = self.env.reset()
                episodes +=1
                ep_lengths.append(episode_steps)
                episode_steps = 0
            if show_progress and step_num % 10 == 0:
                pbar.update(step_num)
        ep_lengths.append(episode_steps)
        return ep_reward/episodes, sum(ep_lengths)/len(ep_lengths)

    def output_report(self):
        f,ax = plt.subplots(2,1,figsize=(10,15))
        ax[0].plot(self.rews)
        ax[0].set_title('Average episode Reward')
        ax[1].plot(self.lens)
        ax[1].set_title('Average Episode Length')
        # ax[2].plot(self.losses)
        # ax[2].set_title('Loss')
        plt.savefig('progress.png')
        plt.close()

    def train(self, n_epochs):
        initial_steps = max(self.min_steps_learn, self.batch_size)
        print('collecting decorrelation steps')
        self.collect_batch(initial_steps,epsilon = 1, show_progress=True)
        self.rews = []
        self.lens = []
        # self.losses = []
        for ep in tqdm(range(n_epochs)):
            avg_rew, avg_len = self.collect_batch(self.steps_per_epoch)
            self.learn()
            self.rews.append(avg_rew)
            self.lens.append(avg_len)
            # self.losses.append(loss)

            if ep % self.report_interval == 0:
                self.output_report()

            if ep % self.target_update_interval:
                self._update_target()


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    dqn = DQN(env)
    dqn.train(400)
    
