import numpy as np
"""Neural network with hidden layer perfoming binary classification """


class NeuralNetwork:
    """Neural Network class with one hidden layer,
                 performing binary classification"""

    def __init__(self, nx, nodes):
        """Class constructor

        Args:
            nx (int): Number of input features.
            nodes (int): Number of nodes in the hidden layer.

        Raises:
            TypeError: If nx or nodes is not an integer.
            ValueError: If nx or nodes is less than 1.

        Attributes:
            __W1: The weights vector for the hidden layer,
                  initialized using a random normal distribution.
            __b1: The bias for the hidden layer, initialized to 0.
            __A1: The activated output for the hidden layer, initialized to 0.
            __W2: The weights vector for the output neuron,
                  initialized using a random normal distribution.
            __b2: The bias for the output neuron, initialized to 0.
            __A2: The activated output for the output neuron (prediction),
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

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter function for __W1"""
        return self.__W1

    @property
    def b1(self):
        """Getter function for __b1"""
        return self.__b1

    @property
    def A1(self):
        """Getter function for __A1"""
        return self.__A1

    @property
    def W2(self):
        """Getter function for __W2"""
        return self.__W2

    @property
    def b2(self):
        """Getter function for __b2"""
        return self.__b2

    @property
    def A2(self):
        """Getter function for __A2"""
        return self.__A2
