from typing import Tuple, List
import gym
from gym import wrappers
import numpy as np
from tensorboard import program
from matplotlib import pyplot as plt
import tensorflow as tf
from tqdm import tqdm

"""
Action Space: { 0:left, 1:down, 2:right, 3:up}
Obs space:    {S: start, G: goal, H:hole (not ok to walk), F:frozen (ok to walk)}
               s = curr_row * n_rows + curr_col (ex: the goal in 4x4 is 3*4+3)
Reward:       {G:1, S,H,F:0}
"""
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
env = gym.wrappers.RecordEpisodeStatistics(env)


# summary_dir = r"G:\My Drive\Master\Year 2\Deep-Reinforcement-Learning\ex1\tb_data"
# summary_writer = tf.summary.create_file_writer(summary_dir)


def _epsilon_greedy_policy(Q_s: np.ndarray,
                           epsilon: float) -> float:
    """
    chooses a random action with prob epsilon and a greedy action with prob. 1-epsilon
    :param Q_s: all state-action value associated with some specific state s
    :param epsilon: sample probability
    :return: an action
    """
    action = env.action_space.sample() if np.random.uniform(0, 1) < epsilon else np.argmax(Q_s)
    return action


def episode_mean_reward(Q_function: np.ndarray, n_episodes: int = 10, n_steps: int = 100) -> float:
    """
    accumulates the reward for the given Q function
    :return: the total reward accumulated given n_episodes and n_steps
    """
    tot_reward = 0
    tot_steps = 0
    for ep in range(n_episodes):
        state = env.reset()
        for step in range(n_steps):
            tot_steps += 1
            action = np.argmax(Q_function[state])  # no exploration here
            state, reward, done, info = env.step(action)
            tot_reward += reward
            if done:
                break
    return tot_reward / tot_steps


def Q_learning(plot_heatmaps: bool,
               n_episodes: int = 5000,
               n_steps: int = 100,
               alpha: float = 0.85,
               gamma: float = 0.99,
               epsilon: float = 0.8,
               ) -> Tuple[np.ndarray, List[float]]:
    """
    epsilon-greedy Q-learning implementation.
        1. We plot the Q-function as heatmaps for given values
        2. After every 100 episodes we evaluate the reward (without exploration)
    :param plot_heatmaps: True iff we wish to plot Q-function heatmaps
    :param n_episodes: number of episodes
    :param n_steps: number of steps per episode
    :param alpha: learning rate
    :param gamma: discount factor
    :param epsilon: exploration probability
    :return: resulted Q function and a list of reward values
    """
    evaluation_reward = []
    Q = np.zeros((env.observation_space.n, env.action_space.n))  # |S| x |A|
    for ep in tqdm(range(n_episodes)):
        state = env.reset()
        if plot_heatmaps:
            plot_when = [499, 1999, n_episodes - 1]
            if ep in plot_when:
                colormap(Q, ep + 1)
        for step in range(n_steps):
            action = _epsilon_greedy_policy(Q[state], epsilon)
            next_state, reward, done, info = env.step(action)
            target = reward + gamma * np.max(Q[next_state])
            Q[state, action] = (1 - alpha) * Q[state, action] + alpha * target
            state = next_state
            if done:
                # env.render()
                break
        if ep % 100 == 0:
            evaluation_reward.append(episode_mean_reward(Q))
    env.close()
    return Q, evaluation_reward


def colormap(Q_func: np.ndarray, episodes: int) -> None:
    """
    plots the Q function as a ~fancy~ colormap
    :param Q_func: |S|x|A| array of Q values
    :param episodes: number of episodes used in the learning process
    """
    Q_func = Q_func.T  # for horizontal display
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 3)
    # axis ticks and labels:
    actions = np.array([a for a in range(env.action_space.n)])
    states = np.array([s for s in range(env.observation_space.n)])
    ax.set_xticks(states), ax.set_yticks(actions, labels=['left(0)', 'down(1)', 'right(2)', 'up(3)'])
    ax.set_ylabel('Action'), ax.set_xlabel('State'), ax.set_title(f'Q function, {episodes} episodes')
    for s in states:
        for a in actions:
            ax.text(s, a, round(Q_func[a, s], 3), ha='center', va='center', color='w')
    # custom grid:
    for s in states:
        ax.axvline(x=s - 0.5, color='w')
    for a in actions:
        ax.axhline(y=a - 0.5, color='w')
    # color-mesh plot:
    c = ax.pcolormesh(states, actions, Q_func, cmap='viridis')
    fig.colorbar(c, ax=ax)
    plt.show()
    plt.savefig(f"Q_function_{episodes}_episodes")


def _check_if_in_colab():
    try:
        import google.colab
        return True
    except ModuleNotFoundError:
        return False


if __name__ == '__main__':
    env.render()
    Q_f, rew_arr = Q_learning(n_episodes=100000, plot_heatmaps=False)
    plt.plot(rew_arr)
    plt.show()
    x = 2
