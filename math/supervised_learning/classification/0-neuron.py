#!/usr/bin/env python3
"""Defines a single neuron performing binary classification"""
import numpy as np

class Neuron:
    """Neuron class"""
    def __init__(self, nx):
        """Class constructor

        Args:
            nx (int): Number of input features to the neuron.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.

        Attributes:
            W: The weights vector for the neuron, initialized using a random normal distribution.
            b: The bias for the neuron, initialized to 0.
            A: The activated output of the neuron (prediction), initialized to 0.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        self.W = np.random.randn(nx).reshape(1, nx)
        self.b = 0
        self.A = 0

