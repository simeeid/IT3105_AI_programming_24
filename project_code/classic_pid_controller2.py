import jax.numpy as jnp

#from abstract_controller import AbstractController

class ClassicPidController():#AbstractController):
    def __init__(self, params):
        self.kp = params[0]
        self.ki = params[1]
        self.kd = params[2]
        self.error_history = jnp.array([0])

    def update_error_history(self, error):
        self.error_history = jnp.append(self.error_history, error)

    def compute_control_signal(self, error, delerror_delt, sum_error):
        return self.kp * error + self.kd * delerror_delt + self.ki * sum_error