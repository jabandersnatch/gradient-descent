import matplotlib.pyplot as plt
import numpy as np


def tanh(x):
    """
    Hyperbolic tangent activation function
    """
    return np.tanh(x)


def tanh_prime(x):
    """
    Derivative of the hyperbolic tangent activation function
    """
    return 1.0 - np.tanh(x) ** 2


def initialize_weights_glorot(n_in, n_out):
    """
    Xavier Glorot weight initialization
    """
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, (n_out, n_in))


class NeuralNetwork:
    """
    A simple feedforward neural network with n hidden layers
    """

    def __init__(self, layer_sizes):
        self.weights = [
            initialize_weights_glorot(x, y)
            for x, y in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        self.velocity_b = [np.zeros(b.shape) for b in self.biases]
        self.velocity_w = [np.zeros(w.shape) for w in self.weights]

    def feedforward(self, a):
        """
        Return the output of the network if 'a' is input.
        """
        activations = [a]
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activations[-1]) + b
            a = tanh(z)
            activations.append(a)
        return activations[-1]

    def backpropagation(self, x, y):
        """
        Return a tuple of gradients for the cost function with respect to biases and weights
        """
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = tanh(z)
            activations.append(activation)

        delta = self.cost_derivative(activations[-1], y) * tanh_prime(zs[-1])
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, len(self.biases) + 1):
            z = zs[-l]
            sp = tanh_prime(z)
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            grad_b[-l] = delta
            grad_w[-l] = np.dot(delta, activations[-l - 1].T)

        return (grad_b, grad_w)

    def gradient_descent(self, mini_batch, eta, mu=0.9):
        """
        Update the network's weights and biases by applying gradient descent using backpropagation to a single mini batch.
        """
        # @TODO: Implement momentum
        pass

    def cost_derivative(self, output_activations, y):
        """
        Return the vector of partial derivatives
        """
        return output_activations - y

    def update_learning_rate(
        self, epoch, initial_lr, schedule_type="step_decay", decay=0.1, drop_every=1000
    ):
        """
        Update the learning rate according to the specified schedule
        """
        if schedule_type == "step_decay":
            return initial_lr * (decay ** np.floor(epoch / drop_every))
        elif schedule_type == "exp_decay":
            return initial_lr * np.exp(-decay * epoch)
        elif schedule_type == "inv_scaling":
            return initial_lr / (1 + decay * epoch)
        else:
            return initial_lr

    def train(
        self,
        training_data,
        epochs,
        mini_batch_size,
        learning_rate,
        decay,
        schedule_type="step_decay",
    ):
        """
        Train the neural network using mini-batch stochastic gradient descent.
        """
        n = len(training_data)
        loss_history = []
        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.gradient_descent(mini_batch, learning_rate)
            learning_rate = self.update_learning_rate(
                j, learning_rate, schedule_type, decay
            )
            current_loss = np.mean(
                [(self.feedforward(x) - y) ** 2 for x, y in training_data]
            )
            loss_history.append(current_loss)
        return loss_history


def main():
    """
    Train a neural network to approximate the sin function
    """
    x = np.arange(0, 2 * np.pi, 0.01)
    y = np.sin(x)
    training_data = [
        (np.array([i]).reshape(1, 1), np.array([j]).reshape(1, 1)) for i, j in zip(x, y)
    ]

    # Initialize the neural network
    nn = NeuralNetwork([1, 5, 1])

    # Plot predictions before training
    initial_predictions = [nn.feedforward(np.array([[i]])) for i in x]
    plt.figure(figsize=(21, 7))
    plt.subplot(1, 3, 1)
    plt.plot(x, y, label="True Sin(x)")
    plt.plot(
        x,
        [p[0, 0] for p in initial_predictions],
        label="Initial NN Predictions",
        linestyle="--",
    )
    plt.title("Before Training")
    plt.xlabel("Input (x)")
    plt.ylabel("Output (Sin(x))")
    plt.legend()
    plt.grid(True)

    # Train the network and get loss history
    loss_history = nn.train(training_data, 2000, 10, 0.05, 0.005)

    # Plot predictions after training
    trained_predictions = [nn.feedforward(np.array([[i]])) for i in x]
    plt.subplot(1, 3, 2)
    plt.plot(x, y, label="True Sin(x)")
    plt.plot(
        x,
        [p[0, 0] for p in trained_predictions],
        label="Trained NN Predictions",
        linestyle="--",
    )
    plt.title("After Training")
    plt.xlabel("Input (x)")
    plt.legend()
    plt.grid(True)

    # Plot loss history
    plt.subplot(1, 3, 3)
    plt.plot(loss_history, label="Training Loss")
    plt.title("Loss During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
