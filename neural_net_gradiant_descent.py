"""
This script presents a simple neural network created with numpy.
The task is to implement the backpropagation algorithm to train
the network to learn the XOR function. The backpropagation algorithm
is based on gradient descent.

Description:
The XOR function is a binary operation that outputs true when the number
of true inputs is odd. The truth table for XOR is as follows:

| x1 | x2 | y |
|----|----|---|
| 0  | 0  | 0 |
| 0  | 1  | 1 |
| 1  | 0  | 1 |
| 1  | 1  | 0 |

Created on Wed April 17 17:07:12 2024
@author: Juan Andrés Méndez Galvis
"""

import numpy as np


def sigmoid(x):
    """
    Sigmoid activation function
    Parameters:
        x: float
    Returns:
        float
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    """
    Derivative of the sigmoid activation function
    Parameters:
        x: float
    Returns:
        float
    """
    return sigmoid(x) * (1 - sigmoid(x))


def initialize_weights_xavier(n_in, n_out):
    """
    Initialize weights using the Xavier initialization
    Description:
    The Xavier initialization is a technique that helps to initialize
    the weights of the neural network in a way that the variance of the
    output of each layer is the same as the variance of the input.
    Parameters:
        n_in: int
        n_out: int
    Returns:
        numpy.ndarray
    """
    return np.random.randn(n_out, n_in) * np.sqrt(1.0 / n_in)


class NeuralNetwork:
    """
    Neural Network class
    Description:
    This class implements a simple neural network with one hidden layer.
    The network uses the sigmoid activation function for the hidden layer
    and the output layer. The network is trained using the backpropagation
    algorithm with gradient descent.
    Parameters:
        layer_sizes: list
        activation: str

    Methods:
        activation(x): float
        activation_prime(x): float
        feedforward(a): float
        backpropagation(x, y): tuple
        gradient_descent(mini_batch, eta, mu): None
        cost_derivative(output_activations, y): float
        train(training_data, epochs, mini_batch_size, learning_rate, decay): None

    Attributes:
        weights: list
        biases: list
        velocity_b: list
        velocity_w: list
    """

    def __init__(self, layer_sizes):
        """
        Initialize the neural network
        """
        self.weights = [
            initialize_weights_xavier(x, y)
            for x, y in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        self.velocity_b = [np.zeros(b.shape) for b in self.biases]
        self.velocity_w = [np.zeros(w.shape) for w in self.weights]

    def feedforward(self, a):
        """
        Feedforward pass
        Description:
        This method computes the feedforward pass of the neural network
        Parameters:
            a: numpy.ndarray
        Returns:
            numpy.ndarray
        """
        activations = [a]
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activations[-1]) + b
            a = sigmoid(z)
            activations.append(a)
        return activations[-1]

    def backpropagation(self, x, y):
        """
        Backpropagation algorithm
        Description:
        This method computes the backpropagation algorithm to train the
        neural network
        Parameters:
            x: numpy.ndarray
            y: numpy.ndarray
        Returns:
            tuple
        """
        # @TODO: Implement the backpropagation algorithm

    def gradient_descent(self, mini_batch, eta, mu=0.9):
        """
        Gradient descent algorithm
        Description:
        This method computes the gradient descent algorithm to train the
        neural network
        Parameters:
            mini_batch: list
            eta: float
            mu: float
        Returns:
            None
        """
        # @TODO: Implement the gradient descent algorithm

    def cost_derivative(self, output_activations, y):
        """
        Cost derivative
        Description:
        This method computes the derivative of the cost function
        Parameters:
            output_activations: numpy.ndarray
            y: numpy.ndarray
        Returns:
            numpy.ndarray
        """
        return output_activations - y

    def train(self, training_data, epochs, mini_batch_size, learning_rate, decay):
        """
        Train the neural network
        Description:
        This method trains the neural network using the backpropagation
        algorithm with gradient descent
        Parameters:
            training_data: list
            epochs: int
            mini_batch_size: int
            learning_rate: float
            decay: float
        Returns:
            None
        """
        n = len(training_data)
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.gradient_descent(mini_batch, learning_rate)
            learning_rate *= decay
            if j % 100 == 0:
                losses = [
                    self.cost_derivative(self.feedforward(x), y) ** 2
                    for x, y in training_data
                ]
                print(f"Epoch {j}, Loss: {np.mean(losses)}")


def main():
    """
    Main function
    Description:
    This function creates a neural network with one hidden layer and trains
    the network to learn the XOR function
    """
    training_data = [
        (np.array([[0], [0]]), np.array([[0]])),
        (np.array([[0], [1]]), np.array([[1]])),
        (np.array([[1], [0]]), np.array([[1]])),
        (np.array([[1], [1]]), np.array([[0]])),
    ]
    nn = NeuralNetwork([2, 2, 1])
    nn.train(
        training_data,
        epochs=20000,
        mini_batch_size=4,
        learning_rate=0.1,
        decay=0.9999,
    )
    for x, y in training_data:
        print(
            f"Input: {x.flatten()} Output: {nn.feedforward(x).flatten()} Target: {y.flatten()}"
        )


if __name__ == "__main__":
    main()
