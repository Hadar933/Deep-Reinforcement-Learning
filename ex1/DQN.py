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
import Config


OPTIMIZERS={
    'Adam': Adam,
    'RMSprop': RMSprop
}


class ExperienceReplay:
    def __init__(self, size: int):
        self.size: int = size
        self._exp_rep: deque = deque([], maxlen=size)

    def append(self, experience):
        self._exp_rep.append(experience)

    def sample(self, batch_size: int):
        rand_sample = random.sample(self._exp_rep, batch_size)
        dict_batch = {
            'states': np.stack([b_step[0] for b_step in rand_sample]),
            'actions': np.array([b_step[1] for b_step in rand_sample]),
            'rewards': np.array([b_step[2] for b_step in rand_sample]),
            'next_states': np.stack([b_step[3] for b_step in rand_sample]),
            'dones': np.array([b_step[4] for b_step in rand_sample])
        }
        return dict_batch

    def __len__(self):
        return self._exp_rep.__len__()

class DQN():

    def __init__(
                self, env : gym.Env, hidden_dims: List[int] = [8,32], lr : float = 0.001 , epsilon_bounds: list[float,float] = [0.3, 0.01], gamma : float = 0.95, 
                learning_epochs: int = 5, batch_size : int = 32, target_update_interval: int =10, steps_per_epoch: int = 256, 
                buffer_size : int =5000, min_steps_learn: int = 0, inner_activation: str = 'relu', verbose : Union[str,int] = 0, 
                final_activation : str = 'linear', optimizer_name: str = 'RMSprop' , loss_fn_name : str = 'mse', 
                kernel_initializer: str = 'glorot_uniform', report_interval = 1):
        assert optimizer_name in OPTIMIZERS.keys() ; "Unknown optimizer"
        self.env = env
        self.action_space   = env.action_space.n
        self.state_space    = env.observation_space.shape[0]
        self.steps_per_epoch = steps_per_epoch
        self.hidden_dims = hidden_dims
        self.target_update_interval = target_update_interval
        self.epsilon = epsilon_bounds[0]
        self.epsilon_bounds = epsilon_bounds
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

    def _update_eps(self):
        self.epsilon = self.epsilon_bounds[0]- (self.epsilon_bounds[0]-self.epsilon_bounds[1])*((self.epoch+1)/self.n_epochs)
        print(self.epsilon)

    def get_action(self, state, epsilon = None):
        epsilon  = self.epsilon if epsilon is None else epsilon
        if epsilon > random.random():
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.q(state))
        return action

    def learn(self):
        batch = self.replay_buffer.sample(self.batch_size)
        gamma = (1 - batch['dones']) * self.gamma
        y = batch['rewards'] + gamma * np.max(self.q_target(batch['next_states']),axis=1)
        y_q = self.q.predict(batch['states'],verbose = 0)                                      # Predict Qs on all actions
        y_q[np.arange(len(y_q)).tolist(),batch['actions'].astype(int).tolist()]=y       # Change the values of the actual actions to target (y)
        loss = self.q.fit(batch['states'],y_q, batch_size = self.batch_size, verbose = 0).history['loss']                                            # loss != 0 only on actual actions takes
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
            action = self.get_action(np.expand_dims(state, 0),epsilon)
            next_state, reward, done, info = self.env.step(action)
            episode_steps +=1
            # if done:
            #     reward = -10
            self.replay_buffer.append([state, action, reward, next_state, done])
            if done:
                state = self.env.reset()
                episodes +=1
                ep_lengths.append(episode_steps)
                episode_steps = 0
            ep_reward += reward
            
            if show_progress and step_num % 10 == 0:
                pbar.update(10)
        if show_progress:
            pbar.close()
        ep_lengths.append(episode_steps)
        return ep_reward/episodes, sum(ep_lengths)/len(ep_lengths)

    def output_report(self):
        f,ax = plt.subplots(1,1,figsize=(10,10))
        ax.plot(self.rews)
        ax.set_title('Average episode Reward')
        # ax[1].plot(self.lens)
        # ax[1].set_title('Average Episode Length')
        plt.savefig('progress.png')
        plt.close('all')

    def train(self, n_epochs):
        # initial_steps = max(self.min_steps_learn, self.batch_size)
        print('collecting decorrelation steps')
        avg_rew, avg_len = self.collect_batch(self.min_steps_learn,epsilon = 1, show_progress=True)
        self.rews = [avg_rew]
        self.lens = [avg_len]
        self.output_report()
        self.n_epochs = n_epochs
        print(f'Training for {n_epochs} epochs')
        for ep in tqdm(range(n_epochs)):
            self.epoch = ep
            self._update_eps()
            avg_rew, avg_len = self.collect_batch(self.steps_per_epoch)
            self.learn()
            self.rews.append(avg_rew)
            self.lens.append(avg_len)
            if ep % self.report_interval == 0:
                self.output_report()
                # aaa=1
            if ep % self.target_update_interval:
                self._update_target()


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    dqn = DQN(env)
    dqn.train(500)
    
