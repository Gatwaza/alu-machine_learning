#!/usr/bin/env python3
"""Defines a single neuron performing binary classification"""

import numpy as np


class Neuron:
    """Neuron class performing binary classification"""

    def __init__(self, nx):
        """Class constructor

        Args:
            nx (int): Number of input features to the neuron.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.

        Attributes:
            __W: The weights vector for the neuron,
                initialized using a random normal distribution.
            __b: The bias for the neuron, initialized to 0.
            __A: The activated output of the neuron (prediction),
                initialized to 0.
        """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be positive")

        self.__W = np.random.randn(nx).reshape(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter function for __W"""
        return self.__W

    @property
    def b(self):
        """Getter function for __b"""
        return self.__b

    @property
    def A(self):
        """Getter function for __A"""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m),
                               where nx is the number of input features
                               and m is the number of examples.

        Returns:
            numpy.ndarray: The activated output of the neuron.
        """

        Z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression.

        Args:
            Y (numpy.ndarray): Correct labels with shape (1, m).
            A (numpy.ndarray): Activated output with shape (1, m).

        Returns:
            float: The cost of the model.
        """

        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost


if __name__ == "__main__":
    # Testing the Neuron class
    neuron = Neuron(5)
    X = np.array([[0.1, 0.2, 0.3],
                  [0.4, 0.5, 0.6],
                  [0.7, 0.8, 0.9],
                  [0.2, 0.3, 0.4],
                  [0.5, 0.6, 0.7]])
    Y = np.array([[0, 1, 0]])
    A = neuron.forward_prop(X)
    print(neuron.cost(Y, A))
