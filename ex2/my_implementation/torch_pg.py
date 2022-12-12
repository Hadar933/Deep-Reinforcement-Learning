import os
import shutil
from typing import List
import torch.nn.functional as F
import gym
import numpy as np
import torch
from gym.envs.classic_control import CartPoleEnv
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

np.random.seed(1)


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


def reinforcetst(env: CartPoleEnv,
                 policy: nn.Module, optimizer: optim.Optimizer,
                 n_episodes: int, gamma: float):
    def init_dict():
        return {'states': [], 'actions': [], 'rewards': []}

    possible_actions = np.arange(env.action_space.n)
    total_rewards = []
    for ep in range(n_episodes):
        history = init_dict()
        state = env.reset()
        done = False
        while not done:
            action_prob = policy(state).detach().numpy()
            action = np.random.choice(possible_actions, p=action_prob)
            next_state, reward, done, info = env.step(action)
            history['states'].append(state), history['actions'].append(action), history['rewards'].append(reward)
            state = next_state

        tot_ep_rew = sum(history['rewards'])
        total_rewards.append(tot_ep_rew)
        TB_WRITER.add_scalar("Reward/ReinforceTER", tot_ep_rew, ep)

        state_tensor = torch.FloatTensor(history['states'])
        reward_tensor = torch.FloatTensor(discounted_return(gamma, history['rewards']).copy())
        action_tensor = torch.LongTensor(history['actions'])
        optimizer.zero_grad()
        log_policy = torch.log(policy(state_tensor))
        # we only choose the log policy that corresponds to actions that we chose
        # log_policy: [|B|,2], index = [|B|,1] -> gathered log_policy [|B|,1]
        gathered_log_policy = torch.gather(log_policy, 1, action_tensor[..., np.newaxis])
        loss = torch.mean(- reward_tensor * gathered_log_policy.squeeze(1))
        loss.backward()
        optimizer.step()

        avg_reward_last_100_epi = np.mean(total_rewards[-100:])
        print(f"Episode [{ep}/{n_episodes}] -- avg reward: {avg_reward_last_100_epi:.2f}", end='\r')


def reinforce(env: CartPoleEnv,
              policy: nn.Module, optimizer: optim.Optimizer,
              n_episodes: int, n_batches: int, gamma: float):
    def init_dict():
        return {'states': [], 'actions': [], 'rewards': []}

    possible_actions = np.arange(env.action_space.n)
    curr_episode = 1
    total_rewards = []
    batched_history = init_dict()
    for ep in range(n_episodes):
        history = init_dict()
        state = env.reset()
        done = False
        while not done:
            action_prob = policy(state).detach().numpy()
            action = np.random.choice(possible_actions, p=action_prob)
            next_state, reward, done, info = env.step(action)
            history['states'].append(state), history['actions'].append(action), history['rewards'].append(reward)
            state = next_state

            if done:
                batched_history['states'].extend(history['states'])
                batched_history['actions'].extend(history['actions'])
                batched_history['rewards'].extend(discounted_return(gamma, history['rewards']))
                tot_ep_rew = sum(history['rewards'])
                total_rewards.append(tot_ep_rew)
                TB_WRITER.add_scalar("Reward/ReinforceTER", tot_ep_rew, ep)
                curr_episode += 1
                if curr_episode == n_batches:
                    optimizer.zero_grad()
                    state_tensor = torch.FloatTensor(batched_history['states'])
                    reward_tensor = torch.FloatTensor(batched_history['rewards'])
                    action_tensor = torch.LongTensor(batched_history['actions'])

                    log_policy = torch.log(policy(state_tensor))
                    # we only choose the log policy that corresponds to actions that we chose
                    # log_policy: [|B|,2], index = [|B|,1] -> gathered log_policy [|B|,1]
                    gathered_log_policy = torch.gather(log_policy, 1, action_tensor[..., np.newaxis])
                    loss = torch.mean(- reward_tensor * gathered_log_policy.squeeze(1))
                    loss.backward()
                    optimizer.step()

                    batched_history = init_dict()
                    curr_episode = 1

        avg_reward_last_100_epi = np.mean(total_rewards[-100:])
        print(f"Episode [{ep}/{n_episodes}] -- avg reward: {avg_reward_last_100_epi:.2f}", end='\r')


def reinforce_with_baseline(env: CartPoleEnv,
                            policy_net: nn.Module, value_net: nn.Module,
                            policy_opt: optim.Optimizer, value_opt: optim.Optimizer,
                            n_episodes: int, n_iters: int, gamma: float):
    def init_dict():
        return {'states': [], 'actions': [], 'rewards': []}

    possible_actions = np.arange(env.action_space.n)
    curr_episode = 1
    total_rewards = []
    batched_history = init_dict()
    I = 1
    for ep in range(n_episodes):
        I *= gamma
        history = init_dict()
        state = env.reset()
        done = False
        while not done:
            action_prob = policy_net(state).detach().numpy()
            action = np.random.choice(possible_actions, p=action_prob)
            next_state, reward, done, info = env.step(action)
            history['states'].append(state), history['actions'].append(action), history['rewards'].append(reward)
            state = next_state

            if done:
                batched_history['states'].extend(history['states'])
                batched_history['actions'].extend(history['actions'])
                batched_history['rewards'].extend(discounted_return(gamma, history['rewards']))
                tot_ep_rew = sum(history['rewards'])
                total_rewards.append(tot_ep_rew)
                TB_WRITER.add_scalar("Reward/ReinforceWBaselineTER", tot_ep_rew, ep)
                curr_episode += 1
                if curr_episode == n_iters:
                    policy_opt.zero_grad()
                    value_opt.zero_grad()
                    state_tensor = torch.FloatTensor(batched_history['states'])
                    reward_tensor = torch.FloatTensor(batched_history['rewards'])
                    action_tensor = torch.LongTensor(batched_history['actions'])

                    value_s = value_net(state_tensor).squeeze()
                    with torch.no_grad():
                        advantage = reward_tensor - value_s
                    log_policy = torch.log(policy_net(state_tensor))
                    gathered_log_policy = torch.gather(log_policy, 1, action_tensor[..., np.newaxis])
                    policy_loss = I * torch.mean(- advantage * gathered_log_policy.squeeze())
                    policy_loss.backward()
                    policy_opt.step()

                    # vf_loss = F.mse_loss(value_s, reward_tensor)
                    vf_loss = I * torch.mean(-advantage * value_s)
                    vf_loss.backward()
                    value_opt.step()

                    batched_history = init_dict()
                    curr_episode = 1

        avg_reward_last_100_epi = np.mean(total_rewards[-100:])
        print(f"Episode [{ep}/{n_episodes}] -- avg reward: {avg_reward_last_100_epi:.2f}", end='\r')


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
    hidden_layer_dims = [64, 64]

    pi_net = PolicyNetwork(hidden_layer_dims, pr_dropout, cart_pole)
    v_net = ValueFunctionNetwork(hidden_layer_dims, pr_dropout, cart_pole)

    gamma_discount = 0.99
    batch_iters = 10
    num_episodes = 10000
    REINFORCE_PI_LR = 0.001
    A2C_PI_LR = 0.0004
    A2C_V_LR = 0.01
    pi_opt = optim.Adam(pi_net.layers.parameters(), lr=REINFORCE_PI_LR)
    v_opt = optim.Adam(v_net.layers.parameters(), lr=A2C_V_LR)

    print("REINFORCE")
    reinforcetst(cart_pole, pi_net, pi_opt, num_episodes, gamma_discount)
    print("REINFORCE w/Baseline")
    # reinforce_with_baseline(cart_pole, pi_net, v_net, pi_opt, v_opt, num_episodes, 20, gamma_discount)
    print("Actor Critic")
    # A2C(cart_pole, pi_net, v_net, pi_opt, v_opt, num_episodes, 1000, gamma_discount)

    TB_WRITER.flush()
    TB_WRITER.close()
