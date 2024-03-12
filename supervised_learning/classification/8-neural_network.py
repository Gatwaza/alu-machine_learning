#!/usr/bin/env python3
"""Neural network with hidden layer """

import numpy as np


class NeuralNetwork:
    """Neural Network class with one hidden,
         layer performing binary classification"""

    def __init__(self, nx, nodes):
        """Class constructor

        Args:
            nx (int): Number of input features.
            nodes (int): Number of nodes in the hidden layer.

        Raises:
            TypeError: If nx or nodes is not an integer.
            ValueError: If nx or nodes is less than 1.

        Attributes:
            W1: The weights vector for the hidden layer,
                initialized using a random normal distribution.
            b1: The bias for the hidden layer, initialized to 0.
            A1: The activated output for the hidden layer,
                 initialized to 0.
            W2: The weights vector for the output neuron,
                initialized using a random normal distribution.
            b2: The bias for the output neuron, initialized to 0.
            A2: The activated output for the output neuron (prediction),
                initialized to 0.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
