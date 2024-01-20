
#from abstract_controller import AbstractController
import numpy as np
import jax
import jax.numpy as jnp

class NeuralNetController():
    def __init__(self, params, num_layers, num_neurons, activation):
        # num layers is the number of hidden layers, in range 0-5
        # num neurons is an array with the number of neurons in each hidden layer

        # the network will have 3 inputs and 1 output
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.activation = activation

        # self.weights = weights
        # self.biases = biases
        self.params = params

        self.error_history = jnp.array([0])

    def update_error_history(self, error):
        self.error_history = jnp.append(self.error_history, error)

    def compute_control_signal(self, all_params, features):
        def sigmoid(x): return 1 / (1 + jnp.exp(-x))
        def tanh(x): return jnp.tanh(x)
        def relu(x): return jnp.maximum(x, 0)

        activations = features
        for weights, biases in all_params:
            activations = sigmoid(jnp.dot(activations, weights) + biases) # will add cases
        return activations
        
    
    def forward_pass(self, error_arr, weights, biases): # where error_arr contains E, dE_dt, sumE: the three inputs
        for i in range(self.num_layers):
            if i == 0:
                layer = jnp.dot(error_arr, weights[i]) + biases[i]
            else:
                layer = jnp.dot(layer, weights[i]) + biases[i]
            if self.activation == "relu":
                layer = jnp.maximum(layer, 0)
            elif self.activation == "sigmoid":
                layer = 1 / (1 + jnp.exp(-layer))
            elif self.activation == "tanh":
                layer = jnp.tanh(layer)
        return layer

    def update_controller(self, error, delerror_delt, sumerror): # to be filled later
        pass

# if __name__ == "__main__":
#     num_layers = 1
#     num_neurons = [3]
#     activation = "relu"
#     range_initial_value = 0.1
#     learning_rate = 0.1

#     neural_net_controller = NeuralNetController(num_layers, num_neurons, activation, range_initial_value)
#     print(neural_net_controller.weights)
#     print(neural_net_controller.biases)