#!/usr/bin/env python3
"""Module for the SARSA(lambda) algorithm for Q-value estimation."""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """Perform the SARSA(lambda) algorithm to estimate Q values.

    Args:
        env: openAI environment instance
        Q: numpy.ndarray of shape (s, a) containing the Q table
        lambtha: eligibility trace factor
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
        epsilon: initial threshold for epsilon greedy
        min_epsilon: minimum value that epsilon should decay to
        epsilon_decay: decay rate for updating epsilon between episodes

    Returns:
        Q: updated Q table
    """
    n_states, n_actions = Q.shape

    for _ in range(episodes):
        state = env.reset()
        Et = np.zeros((n_states, n_actions))

        if np.random.uniform() > epsilon:
            action = np.argmax(Q[state])
        else:
            action = env.action_space.sample()

        for _ in range(max_steps):
            next_state, reward, done, _ = env.step(action)

            if np.random.uniform() > epsilon:
                next_action = np.argmax(Q[next_state])
            else:
                next_action = env.action_space.sample()

            delta = (reward + gamma * Q[next_state, next_action]
                     - Q[state, action])
            Et[state, action] += 1

            Q = Q + alpha * delta * Et
            Et = gamma * lambtha * Et

            state = next_state
            action = next_action

            if done:
                break

        epsilon = max(min_epsilon, epsilon - epsilon_decay)

    return Q
