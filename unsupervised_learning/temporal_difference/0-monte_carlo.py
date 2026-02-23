#!/usr/bin/env python3
"""Module for the Monte Carlo algorithm for value estimation."""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.9):
    """Perform the Monte Carlo algorithm to estimate state values.

    Args:
        env: openAI environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes a state and returns the next action
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V: updated value estimate
    """
    for _ in range(episodes):
        state = env.reset()
        episode = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, reward))
            state = next_state
            if done:
                break

        G = 0
        visited = set()
        for t in range(len(episode) - 1, -1, -1):
            state, reward = episode[t]
            G = gamma * G + reward
            if state not in visited:
                visited.add(state)
                V[state] = V[state] + alpha * (G - V[state])

    return V
