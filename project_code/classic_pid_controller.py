import jax.numpy as jnp

class ClassicPidController():
    """
    A class representing the classic PID controller.

    Attributes:
        kp (float): The proportional gain.
        ki (float): The integral gain.
        kd (float): The derivative gain.
        error_history (ndarray): A history of past errors.

    Methods:
        update_error_history(error): Updates the error history with a new error value.
        compute_control_signal(error, delerror_delt, sum_error): Computes the control signal based on the error, derivative of error, and sum of errors.
    """

    def __init__(self, params):
        self.kp = params[0]
        self.ki = params[1]
        self.kd = params[2]
        self.error_history = jnp.array([0])

    def update_error_history(self, error):
        # to keep track of the error made by the controller
        self.error_history = jnp.append(self.error_history, error)

    def compute_control_signal(self, error, delerror_delt, sum_error):
        # compute the control signal used to control the plant
        return self.kp * error + self.kd * delerror_delt + self.ki * sum_error