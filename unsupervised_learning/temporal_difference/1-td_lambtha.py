#!/usr/bin/env python3
"""Module for the TD(lambda) algorithm for value estimation."""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """Perform the TD(lambda) algorithm to estimate state values.

    Args:
        env: openAI environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes a state and returns the next action
        lambtha: eligibility trace factor
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V: updated value estimate
    """
    n_states = V.shape[0]

    for _ in range(episodes):
        state = env.reset()
        Et = np.zeros(n_states)

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)

            delta = reward + gamma * V[next_state] - V[state]
            Et[state] += 1

            V = V + alpha * delta * Et
            Et = gamma * lambtha * Et

            state = next_state
            if done:
                break

    return V
