import random
import gym
import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop, SGD
import tensorflow as tf

from collections import deque
from typing import Tuple, List, Union
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import Config
import argparse
import inspect

OPTIMIZERS={
    'Adam': Adam,
    'RMSprop': RMSprop,
    'SGD': SGD
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
                self, env : gym.Env, hidden_dims: List[int] = [32,32,32], lr : float = 0.001 , epsilon_bounds: list[float,float] = [1.0, 0.01], eps_decay_fraction: float = 0.1, gamma : float = 0.95, 
                learning_epochs: int = 5, batch_size : int = 32, target_update_interval: int =50, steps_per_epoch: int = 500, 
                buffer_size : int =10000, min_steps_learn: int = 10000, inner_activation: str = 'relu', verbose : Union[str,int] = 0, 
                final_activation : str = 'relu', optimizer_name: str = 'SGD' , loss_fn_name : str = 'mse', dropout: float = 0.1, batch_norm: bool = False,
                kernel_initializer: str = 'he_normal', report_interval: int = 1, save_interval:int = 500):
        assert optimizer_name in OPTIMIZERS.keys() ; "Unknown optimizer"
        self.env = env
        self.action_space   = env.action_space.n
        self.state_space    = env.observation_space.shape[0]
        self.steps_per_epoch = steps_per_epoch
        self.hidden_dims = hidden_dims
        self.target_update_interval = target_update_interval
        self.epsilon = epsilon_bounds[0]
        self.epsilons = []
        self.epsilon_bounds = epsilon_bounds
        self.eps_decay_fraction = eps_decay_fraction
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
        self.save_interval = save_interval
        self.dropout = dropout
        self.bn = batch_norm
        self.q = self._build_model()
        self.q_target = self._build_model()
        self._setup_tensorboard()

        self.ckpt = tf.train.Checkpoint(step = tf.Variable(1),q = self.q,target = self.q_target)
        self.ckpt_mgr = tf.train.CheckpointManager(self.ckpt, './tf_ckpts', max_to_keep=3)

        self.q_updates = []



    def _build_model(self):
        net = Sequential()
        net.add(Dense(self.hidden_dims[0], input_dim=self.state_space, activation=self.inner_act))
        for next_dim in self.hidden_dims[1:]:
            net.add(Dense(next_dim, activation=self.inner_act, kernel_initializer=self.kernel_initializer))
            net.add(Dropout(rate= self.dropout))
            if self.bn:
                net.add(BatchNormalization())
        net.add(Dense(self.env.action_space.n, activation=self.final_activation, kernel_initializer=self.kernel_initializer))
        net.compile(loss=self.loss_fn_name, optimizer=OPTIMIZERS[self.optimizer_name]())
        return net
    
    def _save_model(self):
        self.ckpt_mgr.save()

    def _setup_tensorboard(self):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        self.summary_writer = tf.summary.create_file_writer(train_log_dir)

    def _update_target(self):
        self.q_target.set_weights(self.q.get_weights())

    def _update_eps(self):
        # self.epsilon = 0.99 * self.epsilon
        if (self.epoch+1)/self.n_epochs < self.eps_decay_fraction:
            final_decay_episode = int(self.n_epochs*self.eps_decay_fraction)
            self.epsilon = self.epsilon_bounds[0]- (self.epsilon_bounds[0]-self.epsilon_bounds[1])*((self.epoch+1)/final_decay_episode)
        self.epsilons.append(self.epsilon)

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
        y_q = self.q(batch['states']).numpy()                                     # Predict Qs on all actions
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
        for step_num in range(2*n_steps): # larger than n_steps to make sure we finish the episodes, but no too large so infinite episodes will not result in infinite loops
            action = self.get_action(np.expand_dims(state, 0),epsilon)
            next_state, reward, done, info = self.env.step(action)
            if done:
                reward = -1
            episode_steps +=1
            self.replay_buffer.append([state, action, reward, next_state, done])
            state = next_state
            if done:
                state = self.env.reset()
                episodes +=1
                ep_lengths.append(episode_steps)
                episode_steps = 0
                if step_num >= n_steps:
                    break
            ep_reward += reward
            if show_progress and step_num % 10 == 0:
                pbar.update(10)
        if show_progress:
            pbar.close()
        ep_lengths.append(episode_steps)
        return ep_reward/episodes, sum(ep_lengths)/len(ep_lengths)

    def output_report(self):
        f,ax = plt.subplots(2,2,figsize=(10,10))
        ax=ax.ravel()
        ax[0].plot(self.rews)
        ax[0].set_title('Average episode Reward')
        ax[1].plot(self.losses)
        ax[1].set_title('Training Loss')
        ax[2].plot(self.epsilons)
        ax[2].set_title('Epsilon')
        plt.savefig('progress.png')
        plt.close('all')

    def train(self, n_epochs):
        # initial_steps = max(self.min_steps_learn, self.batch_size)
        self.ckpt.step.assign_add(1)
        print('collecting decorrelation steps')
        avg_rew, avg_len = self.collect_batch(self.min_steps_learn,epsilon = 1, show_progress=True)
        self.rews = [avg_rew]
        self.lens = [avg_len]
        self.losses = []
        self.output_report()
        self.n_epochs = n_epochs
        print(f'Training for {n_epochs} epochs')
        for ep in tqdm(range(n_epochs)):
            self.epoch = ep
            self._update_eps()
            avg_rew, avg_len = self.collect_batch(self.steps_per_epoch)
            loss = self.learn()
            self.rews.append(avg_rew)
            self.lens.append(avg_len)
            self.losses.append(loss)
            with self.summary_writer.as_default():
                tf.summary.scalar('loss', loss[0], step=ep)
                tf.summary.scalar('Avg_reward', avg_rew, step=ep)
                tf.summary.scalar('Avg_len', avg_len, step=ep)
                tf.summary.scalar('Epsilon', self.epsilon, step=ep)
            if ep % self.target_update_interval==0:
                self._update_target()
            if ep % self.save_interval ==0:
                self._save_model()


def parse_args():
    fn_args = inspect.get_annotations(DQN.__init__)
    signature = inspect.signature(DQN.__init__)

    args = {k: (fn_args[k], v.default) for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}
    parser = argparse.ArgumentParser(description='DQN implementation in TF baaaaa')

    for arg in args.keys():
        parser.add_argument(f'--{arg}',type=args[arg][0], default=args[arg][1], required=False)
    args = vars(parser.parse_args())

if __name__ == '__main__':
    parse_args()
    env = gym.make('CartPole-v1')
    dqn = DQN(env)
    dqn.train(10000)
    
