# from abstract_controller import AbstractController

import numpy as np
import jax
import jax.numpy as jnp

# create a neural network with three inputs and one output using forward pass and backward propagation using jax
class NeuralNetController():
    def __init__(self):
        num_layers = 1
        num_neurons = [3]
        activation = "relu"
        range_initial_value = 0.1
        num_epochs = 100
        learning_rate = 0.1

        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.activation = activation
        self.range_initial_value = range_initial_value
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.weights = []
        self.biases = []

        self.initialize_weights_and_biases()

    def initialize_weights_and_biases(self):
        for i in range(self.num_layers):
            if i == 0:
                self.weights.append(self.range_initial_value * np.random.randn(3, self.num_neurons[i]))
                self.biases.append(self.range_initial_value * np.random.randn(1, self.num_neurons[i]))
            else:
                self.weights.append(self.range_initial_value * np.random.randn(self.num_neurons[i - 1], self.num_neurons[i]))
                self.biases.append(self.range_initial_value * np.random.randn(1, self.num_neurons[i]))
        # since output is only one number, we need to add one more weight and bias
        self.weights.append(self.range_initial_value * np.random.randn(self.num_neurons[-1], 1))
        #self.biases.append(self.range_initial_value * np.random.randn(1, 1))
        self.biases.append(1.0)

    # ==================== compute_control_signal ====================

    def compute_control_signal(self, error, delerror_delt, sumerror):
        error_arr = jnp.array([error, delerror_delt, sumerror])
        return self.forward_pass(error_arr, self.weights, self.biases)

    def forward_pass(self, error_arr, weights, biases): # where error_arr contains E, dE_dt, sumE: the three inputs
        for i in range(self.num_layers):
            if i == 0:
                layer = jnp.dot(error_arr, weights[i]) + biases[i]
            else:
                layer = jnp.dot(layer, weights[i]) + biases[i]
            if self.activation == "relu":
                layer = jax.nn.relu(layer)
            elif self.activation == "sigmoid":
                layer = jax.nn.sigmoid(layer)
            elif self.activation == "tanh":
                layer = jax.nn.tanh(layer)
        # return layer
        return jnp.dot(layer, self.weights[-1]) + self.biases[-1]
    
    # ==================== update_controller ====================
    
    def update_controller(self, prediction_error):
        self.backward_propagation(prediction_error)
    
    def backward_propagation(self, prediction_error): # where prediction_error is an array of the errors between the plant output and the desired output in an epoch
        """ for j in range(len(prediction_error)):
            delmse_delomega = jax.grad(self.compute_mse(prediction_error[j]))([self.weights, self.biases])
            self.update(delmse_delomega[0], delmse_delomega[1]) """
        print("mse: ", self.compute_mse(0, 0, 0, self.weights, self.biases, 1))
        #delmse_delomega = jax.grad(self.compute_mse(prediction_error))([self.weights, self.biases])
        # think maybe I have to include the weights and biases in the mse function so that it can be differentiated
        #delmse_delomega = jax.grad(self.compute_mse, argnums=[1, 2]) # if weights and bias are arg 2 and 3
        delmse_delomega = jax.grad(self.compute_mse, argnums=[3, 4])
        print("weights: ", self.weights)
        print("biases: ", self.biases)
        self.update(delmse_delomega(0,0,0,self.weights,self.biases,1)[0], delmse_delomega(0,0,0,self.weights,self.biases,1)[1])
        print("weights: ", self.weights)
        print("biases: ", self.biases)

    # def compute_mse(self, prediction_error):
    #     return jnp.mean(jnp.square(prediction_error))
        
    def compute_mse(self, error, delerror_delt, sum_error, weights, biases, target):
        error_arr = jnp.array([error, delerror_delt, sum_error])
        return jnp.mean(jnp.square(target - self.forward_pass(error_arr, weights, biases)))
        

    def update(self, delmse_delomega_weights, delmse_delomega_biases):
        for i in range(self.num_layers):
            self.weights[i] -= self.learning_rate * delmse_delomega_weights[i]
            self.biases[i] -= self.learning_rate * delmse_delomega_biases[i]

nn1 = NeuralNetController()
#for _ in range(5):
for _ in range(5):
    pred1 = nn1.compute_control_signal(0,0,0)
    #print(pred1)
    nn1.update_controller(pred1)