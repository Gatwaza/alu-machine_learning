#!/usr/bin/env python3
"""Neural network with gradient descent"""

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

    def sigmoid(self, Z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-Z))

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m),
                               where nx is the number of input features
                               and m is the number of examples.

        Returns:
            tuple: A tuple containing two elements:
                   - numpy.ndarray: The activated output for the hidden layer.
                   - numpy.ndarray: The activated output for
                                         the output neuron (prediction).
        """
        Z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = self.sigmoid(Z1)
        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = self.sigmoid(Z2)
        return self.__A1, self.__A2

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

    def evaluate(self, X, Y):
        """Evaluates the neural network's predictions.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels with shape (1, m).

        Returns:
            tuple: A tuple containing two elements:
                   - numpy.ndarray: The predicted labels with shape (1, m).
                   - float: The cost of the network.
        """
        _, A2 = self.forward_prop(X)
        predictions = np.where(A2 >= 0.5, 1, 0)
        cost = self.cost(Y, A2)
        return predictions, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network.

        Args:
            X (numpy.ndarray): Input data with shape (nx, m).
            Y (numpy.ndarray): Correct labels with shape (1, m).
            A1 (numpy.ndarray): Activated output for the hidden layer.
            A2 (numpy.ndarray): Activated output for the
                                         output neuron (prediction).
            alpha (float): The learning rate.

        Updates:
            Updates the private attributes __W1, __b1, __W2, and __b2.
        """
        m = Y.shape[1]

        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dZ1 = np.dot(self.__W2.T, dZ2) * A1 * (1 - A1)
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2
