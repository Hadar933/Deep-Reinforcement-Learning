import os
import shutil
from typing import List
import torch.nn.functional as F
import gym
import numpy as np
import torch
from gym.envs.classic_control import CartPoleEnv
from torch import nn, optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from itertools import count
from torch.autograd import Variable
from collections import deque

np.random.seed(543)
torch.manual_seed(543)


class PolicyNetwork(nn.Module):
    def __init__(self, hidden_dims: List[int], dropout: float, env: CartPoleEnv):
        super().__init__()
        self.state_dim: int = env.observation_space.shape[0]
        self.action_dim: int = env.action_space.n
        self.hidden_dims: List[int] = hidden_dims
        self.dropout = dropout
        self.layers = self._init_layers()

    def _init_layers(self) -> nn.Module:
        net = nn.ModuleList()
        curr_input_dim = self.state_dim
        for hdim in self.hidden_dims:
            net.append(nn.Linear(curr_input_dim, hdim))
            curr_input_dim = hdim
        net.append(nn.Linear(curr_input_dim, self.action_dim))
        return net

    def forward(self, state) -> torch.Tensor:
        state = torch.FloatTensor(state)
        for layer in self.layers[:-1]:
            state = F.relu(layer(state))
            state = F.dropout(state, p=self.dropout)
        action_vec = F.softmax(self.layers[-1](state), dim=-1)
        return action_vec


class ValueFunctionNetwork(nn.Module):
    def __init__(self, hidden_dims: List[int], dropout: float, env: CartPoleEnv):
        super().__init__()
        self.state_dim: int = env.observation_space.shape[0]
        self.output_dim = 1  # one float for the value of every state
        self.hidden_dims: List[int] = hidden_dims
        self.dropout = dropout
        self.layers = self._init_layers()

    def _init_layers(self) -> nn.Module:
        net = nn.ModuleList()
        curr_input_dim = self.state_dim
        for hdim in self.hidden_dims:
            net.append(nn.Linear(curr_input_dim, hdim))
            curr_input_dim = hdim
        net.append(nn.Linear(curr_input_dim, self.output_dim))
        return net

    def forward(self, state) -> torch.Tensor:
        state = torch.FloatTensor(state)
        for layer in self.layers[:-1]:
            state = F.relu(layer(state))
            state = F.dropout(state, p=self.dropout)
        action_vec = F.relu(self.layers[-1](state))
        return action_vec


def discounted_return(gamma: float, rewards: List[float]) -> np.ndarray:
    reward_for_every_t = np.array([gamma ** i * rewards[i] for i in range(len(rewards))])
    G = np.cumsum(reward_for_every_t[::-1])[::-1]
    return G


def reinforce(env: CartPoleEnv,
              policy: nn.Module, optimizer: optim.Optimizer,
              n_episodes: int, gamma: float):
    rewards_arr = []
    eps = np.finfo(np.float32).eps.item()

    for ep in range(n_episodes):
        ep_H = {'states': [], 'actions': [], 'rewards': [], 'log_probs': []}
        state = env.reset()
        ep_tot_rew = 0
        done = False
        while not done:
            state = torch.from_numpy(state).float().unsqueeze(0)
            action_probs = policy(state)
            m = Categorical(action_probs)
            action = m.sample()
            ep_H['log_probs'].append(m.log_prob(action))
            action = action.item()
            state, reward, done, info = env.step(action)
            ep_tot_rew += reward
            ep_H['rewards'].append(reward)

        TB_WRITER.add_scalar("Reward/ReinforceTER", ep_tot_rew, ep)
        returns = torch.tensor(discounted_return(gamma, ep_H['rewards']).copy())
        returns = (returns - returns.mean()) / (returns.std() + eps)
        policy_loss = []
        for log_prob, R in zip(ep_H['log_probs'], returns):
            policy_loss.append(-log_prob * R)
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        rewards_arr.append(ep_tot_rew)
        mean_last_100 = np.mean(rewards_arr[-100:])
        print(f"Episode [{ep}/{n_episodes}] -- avg reward: {mean_last_100:.2f}", end='\r')
        if mean_last_100 >= env.spec.reward_threshold:
            print("Reached env goal!")
            return


def reinforce_with_baseline_tst(env: CartPoleEnv,
                                policy_net: nn.Module, value_net: nn.Module,
                                policy_opt: optim.Optimizer, value_opt: optim.Optimizer,
                                n_episodes: int, n_iters: int, gamma: float):
    rewards_arr = []
    eps = np.finfo(np.float32).eps.item()
    I = 1
    for ep in range(n_episodes):
        I *= gamma
        ep_H = {'states': [], 'actions': [], 'rewards': [], 'log_probs': [], 'state_values': []}
        state = env.reset()
        ep_tot_rew = 0
        done = False
        while not done:
            state = torch.from_numpy(state).float().unsqueeze(0)
            action_probs = policy_net(state)
            m = Categorical(action_probs)
            action = m.sample()
            ep_H['log_probs'].append(m.log_prob(action))
            action = action.item()
            state, reward, done, info = env.step(action)
            ep_tot_rew += reward
            ep_H['rewards'].append(reward)
            state_value = value_net(state)
            ep_H['state_values'].append(state_value)

        TB_WRITER.add_scalar("Reward/ReinforceWBaselineTER", ep_tot_rew, ep)
        returns = torch.tensor(discounted_return(gamma, ep_H['rewards']).copy())
        returns = (returns - returns.mean()) / (returns.std() + eps)
        policy_loss = []
        value_loss = []
        for log_prob, r, s_value in zip(ep_H['log_probs'], returns, ep_H['state_values']):
            with torch.no_grad(): advantage = r - s_value
            policy_loss.append(-advantage * log_prob)
            # value_loss.append(-advantage * s_value)
            value_loss.append(F.smooth_l1_loss(s_value, torch.tensor([r])))

        policy_opt.zero_grad()
        policy_loss = I * torch.cat(policy_loss).sum()
        policy_loss.backward()
        policy_opt.step()

        value_opt.zero_grad()
        value_loss = I * torch.stack(value_loss).sum()
        value_loss.backward()
        value_opt.step()

        rewards_arr.append(ep_tot_rew)
        mean_last_100 = np.mean(rewards_arr[-100:])
        print(f"Episode [{ep}/{n_episodes}] -- avg reward: {mean_last_100:.2f}", end='\r')
        if mean_last_100 >= env.spec.reward_threshold:
            print("Reached env goal!")
            return


def A2C(env: CartPoleEnv,
        policy_net: nn.Module, value_net: nn.Module,
        policy_opt: optim.Optimizer, value_opt: optim.Optimizer,
        n_episodes: int, n_iters: int, gamma: float):
    def ep_history():
        return {'states': [], 'actions': [], 'rewards': [], 'dones': [], 'next_states': []}

    possible_actions = np.arange(env.action_space.n)
    total_rewards = []
    for ep in range(n_episodes):
        I_gamma_multiplier = 1
        H = ep_history()
        state = env.reset()
        for i in range(n_iters):
            action_prob = policy_net(state).detach().numpy()
            action = np.random.choice(possible_actions, p=action_prob)
            next_state, reward, done, info = env.step(action)
            H['states'].append(state), H['actions'].append(action), H['rewards'].append(reward), H['dones'].append(done)
            H['next_states'].append(next_state)
            state = next_state
            if done:
                break
        state_tensor = torch.FloatTensor(H['states'])
        next_state_tensor = torch.FloatTensor(H['next_states'])
        reward_tensor = torch.FloatTensor(H['rewards'])
        action_tensor = torch.LongTensor(H['actions'])
        done_tensor = torch.IntTensor(H['dones'])

        v_s = value_net(state_tensor).squeeze(1)
        v_next_s = value_net(next_state_tensor).squeeze(1)
        with torch.no_grad():
            td_error = reward_tensor + gamma * (1 - done_tensor) * v_next_s - v_s
        log_policy = torch.log(policy_net(state_tensor))
        gathered_log_policy = torch.gather(log_policy, 1, action_tensor[..., np.newaxis])
        policy_loss = I_gamma_multiplier * torch.mean(- td_error * gathered_log_policy.squeeze())
        v_loss = I_gamma_multiplier * torch.mean(- td_error * v_s)

        policy_opt.zero_grad()
        policy_loss.backward()
        policy_opt.step()

        value_opt.zero_grad()
        v_loss.backward()
        value_opt.step()

        I_gamma_multiplier *= gamma
        tot_ep_rew = torch.sum(reward_tensor)
        total_rewards.append(tot_ep_rew)
        TB_WRITER.add_scalar("Reward/A2CTER", tot_ep_rew, ep)
        avg_reward_last_100_epi = np.mean(total_rewards[-100:])
        print(f"Episode [{ep}/{n_episodes}] -- avg reward: {avg_reward_last_100_epi:.2f}", end='\r')


if __name__ == '__main__':
    tb_path = r'G:\My Drive\Master\Year 2\Deep-Reinforcement-Learning\ex2\my_implementation\runs'
    if os.path.exists(tb_path):
        shutil.rmtree(tb_path)
    TB_WRITER = SummaryWriter()

    cart_pole: CartPoleEnv = gym.make('CartPole-v1')
    pr_dropout = 0.6
    hidden_layer_dims = [128]  # for A2C 0-> [64,64]

    pi_net = PolicyNetwork(hidden_layer_dims, pr_dropout, cart_pole)
    v_net = ValueFunctionNetwork(hidden_layer_dims, pr_dropout, cart_pole)

    gamma_discount = 0.99
    num_episodes = 10000
    REINFORCE_PI_LR = 0.01
    A2C_PI_LR = 0.0004
    A2C_V_LR = 0.01
    pi_opt = optim.Adam(pi_net.layers.parameters(), lr=REINFORCE_PI_LR)
    v_opt = optim.Adam(v_net.layers.parameters(), lr=A2C_V_LR)

    print("REINFORCE")
    # reinforce(cart_pole, pi_net, pi_opt, num_episodes, gamma_discount)
    print("REINFORCE w/Baseline")
    # reinforce_with_baseline_tst(cart_pole, pi_net, v_net, pi_opt, v_opt, num_episodes, 20, gamma_discount)
    print("Actor Critic")
    A2C(cart_pole, pi_net, v_net, pi_opt, v_opt, num_episodes, 1000, gamma_discount)

    TB_WRITER.flush()
    TB_WRITER.close()
