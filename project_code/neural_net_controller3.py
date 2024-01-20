
#from abstract_controller import AbstractController
import numpy as np
import jax
import jax.numpy as jnp

class NeuralNetController():
    def __init__(self, params, num_layers, num_neurons, activation):
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.activation = activation

        self.params = params
        self.error_history = jnp.array([0])

    def update_error_history(self, error):
        self.error_history = jnp.append(self.error_history, error)

    def compute_control_signal(self, all_params, features, activation_function):
        def sigmoid(x): return 1 / (1 + jnp.exp(-x))
        def tanh(x): return jnp.tanh(x)
        def relu(x): return jnp.maximum(x, 0)

        if activation_function == "sigmoid":
            activation_function = sigmoid
        elif activation_function == "tanh":
            activation_function = tanh
        elif activation_function == "relu":
            activation_function = relu
        else:
            raise ValueError(f"Activation function \"{activation_function}\" not found")

        activations = features
        for weights, biases in all_params:
            activations = activation_function(jnp.dot(activations, weights) + biases) # will add cases
        return activations
