#!/usr/bin/env python3
"""Module for computing policy using a weight matrix via softmax."""
import numpy as np


def policy(matrix, weight):
    """Compute the policy probabilities given a state matrix and weight matrix.

    Args:
        matrix: numpy.ndarray of shape (1, state_size) representing the state.
        weight: numpy.ndarray of shape (state_size, action_size) representing
                the weight matrix.

    Returns:
        numpy.ndarray of shape (1, action_size) with softmax probabilities
        for each action.
    """
    z = matrix.dot(weight)
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum(axis=1, keepdims=True)
