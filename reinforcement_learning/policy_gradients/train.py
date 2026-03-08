#!/usr/bin/env python3
"""Module implementing the REINFORCE Monte-Carlo policy gradient training."""
import numpy as np
from policy_gradient import policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """Train an agent using the REINFORCE Monte-Carlo policy gradient algorithm.

    Runs nb_episodes of CartPole, collects (state, action, reward) tuples,
    then updates the weight matrix using discounted returns at each timestep.

    Args:
        env: the initial gym environment instance.
        nb_episodes (int): number of episodes used for training.
        alpha (float): the learning rate for weight updates. Default 0.000045.
        gamma (float): the discount factor for future rewards. Default 0.98.

    Returns:
        list: all score values (sum of rewards) for each episode.
    """
    weight = np.random.rand(env.observation_space.shape[0],
                            env.action_space.n)
    scores = []

    for episode in range(nb_episodes):
        state = env.reset()[None, :]
        episode_states = []
        episode_actions = []
        episode_rewards = []

        while True:
            action, grad = policy_gradient(state, weight)
            next_state, reward, done, _ = env.step(action)

            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)

            state = next_state[None, :]
            if done:
                break

        score = sum(episode_rewards)
        scores.append(score)

        for t in range(len(episode_rewards)):
            discounted_return = sum(
                gamma ** (k - t) * episode_rewards[k]
                for k in range(t, len(episode_rewards))
            )
            _, grad = policy_gradient(episode_states[t], weight)
            weight += alpha * grad * discounted_return

        print("Episode: {}/{}, Score: {}".format(
            episode + 1, nb_episodes, score), end="\r", flush=False)

    return scores
