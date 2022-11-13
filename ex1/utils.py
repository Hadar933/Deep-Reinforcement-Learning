from typing import Tuple, List

import numpy as np
from matplotlib import pyplot as plt


def _epsilon_greedy_policy(env, Q_s: np.ndarray,
                           epsilon: float) -> float:
    """
    chooses a random action with prob epsilon and a greedy action with prob. 1-epsilon
    :param Q_s: all state-action value associated with some specific state s
    :param epsilon: sample probability
    :return: an action
    """
    action = env.action_space.sample() if np.random.uniform(0, 1) < epsilon else np.argmax(Q_s)
    return action


def episode_mean_reward(env, Q_function: np.ndarray, n_episodes: int = 10, n_steps: int = 100) -> Tuple[float, float]:
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
    success = 100 * tot_reward / n_episodes
    return tot_reward / tot_steps, success


def colormap(env, Q_func: np.ndarray, steps) -> None:
    """
    plots the Q function as a ~fancy~ colormap
    :param env: gym environment
    :param Q_func: |S|x|A| array of Q values
    :param steps: number of steps used in the learning process
    """
    Q_func = Q_func.T  # for horizontal display
    fig, ax = plt.subplots()
    fig.set_size_inches(12, 4)
    # axis ticks and labels:
    actions = np.array([a for a in range(env.action_space.n)])
    states = np.array([s for s in range(env.observation_space.n)])
    ax.set_xticks(states), ax.set_yticks(actions, labels=['left(0)', 'down(1)', 'right(2)', 'up(3)'])
    ax.set_ylabel('Action'), ax.set_xlabel('State'), ax.set_title(f'Q function, {steps} steps')
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
    # plt.savefig(f"Q_function_{steps}_steps")


def smooth(reward_arr: List[float], smooth_factor: float) -> List[float]:
    """
    smooths the reward values using weighted running average
    :return: smoothed reward
    """
    assert 0 < smooth_factor < 1, 'Smooth factor must be in (0,1)'
    last = reward_arr[0]  # First value in the plot (first timestep)
    smoothed = []
    for point in reward_arr:
        smoothed_val = last * smooth_factor + (1 - smooth_factor) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed
