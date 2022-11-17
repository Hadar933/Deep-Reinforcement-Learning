from typing import Tuple, List
import gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from ex1.utils import _epsilon_greedy_policy, smooth, colormap

"""
Action Space: { 0:left, 1:down, 2:right, 3:up}
Obs space:    {S: start, G: goal, H:hole (not ok to walk), F:frozen (ok to walk)}
               s = curr_row * n_rows + curr_col (ex: the goal in 4x4 is 3*4+3)
Reward:       {G:1, S,H,F:0}
"""
SLIPPERY = False
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=SLIPPERY)
env = gym.wrappers.RecordEpisodeStatistics(env)
GOAL_STATE = 15


def Q_learning(n_episodes: int = 5000,
               n_steps: int = 100,
               alpha: float = 0.25,
               gamma: float = 0.95,
               epsilon: float = 1.0,
               ) -> Tuple[np.ndarray, List[int], List[float]]:
    """
    epsilon-greedy Q-learning implementation.
        1. We plot the Q-function as heatmaps for given values
        2. After every 100 episodes we evaluate the reward (without exploration)
    :param n_episodes: number of episodes
    :param n_steps: number of steps per episode
    :param alpha: learning rate
    :param gamma: discount factor
    :param epsilon: exploration probability (decays over episodes)
    :return: resulted Q function, reward per episode array and average number of steps to goal array
    """
    ep_rew_arr = []
    steps_to_goal_arr_over_100_episodes = []
    mean_steps_to_goal_arr = []
    Q = np.zeros((env.observation_space.n, env.action_space.n))  # |S| x |A|
    tot_steps = 0
    for ep in tqdm(range(n_episodes)):
        ep_rew = 0
        linear_decay_eps = epsilon - (1 / n_episodes)
        state = env.reset()
        for step in range(n_steps):
            tot_steps += 1
            action = _epsilon_greedy_policy(env, Q[state], linear_decay_eps)
            next_state, reward, done, info = env.step(action)
            ep_rew += reward
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


if __name__ == '__main__':
    Q_f, rew, steps_to_goal_arr = Q_learning()
    colormap(env, Q_f, 'all')
    plt.plot(rew, linewidth=0.25, alpha=0.5)
    smooth_weight = 0.8
    plt.plot(smooth(rew, smooth_weight), linewidth=0.25)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Episode Reward (is_slippery={SLIPPERY})')
    plt.grid()
    plt.legend(['Original', f'Smooth {smooth_weight}'])
    # plt.savefig('Episode Reward (is_slippery=True))')
    plt.show()

    plt.plot(range(0, 5000, 100), steps_to_goal_arr)
    plt.xlabel('Episode')
    plt.ylabel('Avg #steps')
    plt.title(f'Avg #steps to the goal over last 100 episodes (is_slippery={SLIPPERY})')
    plt.grid()
    # plt.savefig('Avg #steps to the goal over last 100 episodes (is_slippery=True)')
    plt.show()
