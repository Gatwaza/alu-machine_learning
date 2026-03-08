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

def policy_gradient(state, weight):
    """Compute the Monte-Carlo policy gradient for a state and weight matrix.

    Uses the REINFORCE algorithm: the gradient of log-probability of the
    chosen action with respect to the weight matrix, computed as the outer
    product of the state and (one_hot - probs) — the score function.

    Args:
        state: numpy.ndarray of shape (1, state_size) representing the
               current observation of the environment.
        weight: numpy.ndarray of shape (state_size, action_size) of
                random weights.

    Returns:
        tuple: (action, grad) where
            action (int): the sampled action index.
            grad (numpy.ndarray): gradient of shape (state_size, action_size).
    """
    probs = policy(state, weight)
    action = np.random.choice(probs.shape[1], p=probs[0])
    one_hot = np.zeros(probs.shape[1])
    one_hot[action] = 1
    grad = state.T.dot((one_hot - probs))
    return action, grad
