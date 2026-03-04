#!/usr/bin/env python3
"""Module to have a trained agent play an episode of FrozenLake"""
import numpy as np


def play(env, Q, max_steps=100):
    """Has the trained agent play an episode of FrozenLake.

    Args:
        env: the FrozenLakeEnv instance
        Q: numpy.ndarray containing the Q-table
        max_steps: the maximum number of steps in the episode

    Returns:
        the total rewards for the episode
    """
    state = env.reset()
    total_reward = 0

    for step in range(max_steps):
        env.render()
        action = np.argmax(Q[state])
        state, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            env.render()
            break

    return total_reward
