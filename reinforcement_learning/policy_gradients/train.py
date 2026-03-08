#!/usr/bin/env python3
"""Module implementing the REINFORCE Monte-Carlo policy gradient training."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from policy_gradient import policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """Train an agent using the REINFORCE Monte-Carlo policy gradient algorithm.

    Runs nb_episodes of CartPole, collects (state, action, reward) tuples,
    then updates the weight matrix using discounted returns at each timestep.
    When show_result is True, saves a state-trajectory plot every 1000 episodes
    as 'result_episode_<N>.png' (headless workaround: no OpenGL/display needed).

    Args:
        env: the initial gym environment instance.
        nb_episodes (int): number of episodes used for training.
        alpha (float): the learning rate for weight updates. Default 0.000045.
        gamma (float): the discount factor for future rewards. Default 0.98.
        show_result (bool): if True, save a plot of the episode state trajectory
                            every 1000 episodes. Default False.

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

        render = show_result and ((episode + 1) % 1000 == 0)

        while True:
            action, _ = policy_gradient(state, weight)
            next_state, reward, done, _ = env.step(action)

            episode_states.append(state[0])
            episode_actions.append(action)
            episode_rewards.append(reward)

            state = next_state[None, :]
            if done:
                break

        score = sum(episode_rewards)
        scores.append(score)

        if render:
            states_arr = np.array(episode_states)
            labels = ['Cart Pos', 'Cart Vel', 'Pole Angle', 'Pole Vel']
            fig, axes = plt.subplots(4, 1, figsize=(8, 6))
            fig.suptitle('Episode {} - Score: {}'.format(
                episode + 1, int(score)))
            for i, ax in enumerate(axes):
                ax.plot(states_arr[:, i])
                ax.set_ylabel(labels[i], fontsize=8)
                ax.tick_params(labelsize=7)
            axes[-1].set_xlabel('Timestep')
            plt.tight_layout()
            fig.savefig('result_episode_{}.png'.format(episode + 1))
            plt.close(fig)

        for t in range(len(episode_rewards)):
            discounted_return = sum(
                gamma ** (k - t) * episode_rewards[k]
                for k in range(t, len(episode_rewards))
            )
            _, grad = policy_gradient(episode_states[t][None, :], weight)
            weight += alpha * grad * discounted_return

        print("Episode: {}/{}, Score: {}".format(
            episode + 1, nb_episodes, score), end="\r", flush=False)

    return scores
