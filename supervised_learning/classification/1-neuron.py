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


if __name__ == "__main__":
    import numpy as np

    Neuron = __import__('1-neuron').Neuron

    lib_train = np.load('../data/Binary_Train.npz')
    X_3D, Y = lib_train['X'], lib_train['Y']
    X = X_3D.reshape((X_3D.shape[0], -1)).T

    np.random.seed(0)
    neuron = Neuron(X.shape[0])
    print(neuron.W)
    print(neuron.b)
    print(neuron.A)
    # Note: This line won't change the private attribute __A. It's just
    # setting a new attribute __A.
    neuron.__A = 10
    print(neuron.A)
