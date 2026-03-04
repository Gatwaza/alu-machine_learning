#!/usr/bin/env python3
"""Module to initialize the Q-table for the FrozenLake environment"""
import numpy as np


def q_init(env):
    """Initializes the Q-table for a given FrozenLake environment.

    Args:
        env: the FrozenLakeEnv instance

    Returns:
        the Q-table as a numpy.ndarray of zeros with shape
        (number of states, number of actions)
    """
    return np.zeros((env.observation_space.n, env.action_space.n))
