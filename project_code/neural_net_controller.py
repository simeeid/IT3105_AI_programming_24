
import numpy as np
import jax
import jax.numpy as jnp

class NeuralNetController():
    """
    A class representing the neural network controller.

    Args:
        params (list): List of tuples containing weights and biases for each layer.
        num_layers (int): Number of layers in the neural network.
        num_neurons (int): Number of neurons in each layer.
        activation (str): Activation function to be used in the neural network.

    Attributes:
        num_layers (int): Number of layers in the neural network.
        num_neurons (int): Number of neurons in each layer.
        activation (str): Activation function used in the neural network.
        params (list): List of tuples containing weights and biases for each layer.
        error_history (ndarray): Array containing the history of errors during training.

    Methods:
        update_error_history: Updates the error history with a new error value.
        compute_control_signal: Computes the control signal based on the given features and activation function.
    """

    def __init__(self, params, num_layers, num_neurons, activation):
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.activation = activation

        self.params = params
        self.error_history = jnp.array([0])

    def update_error_history(self, error):
        # to keep track of the error made by the controller
        self.error_history = jnp.append(self.error_history, error)

    def compute_control_signal(self, all_params, features, activation_function):
        # compute the control signal used to control the plant
        # activation function may be chosen for each run
        def sigmoid(x): return 1 / (1 + jnp.exp(-x))
        def tanh(x): return jnp.tanh(x)
        def relu(x): return jnp.maximum(x, 0)
        # soft version of relu
        def soft_relu(x): return jnp.log(1 + jnp.exp(x))

        if activation_function == "sigmoid":
            activation_function = sigmoid
        elif activation_function == "tanh":
            activation_function = tanh
        elif activation_function == "relu":
            activation_function = relu
        elif activation_function == "soft_relu":
            activation_function = soft_relu
        else:
            raise ValueError(f"Activation function \"{activation_function}\" not found")

        activations = features
        for weights, biases in all_params:
            activations = activation_function(jnp.dot(activations, weights) + biases) 
        return activations
