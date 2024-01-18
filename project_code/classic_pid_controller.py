import jax.numpy as jnp

#from abstract_controller import AbstractController

class ClassicPidController():
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error_history = jnp.array([0])

    def update_error_history(self, error):
        self.error_history = jnp.append(self.error_history, error)

    def compute_control_signal(self, error, delerror_delt, sum_error):
        return self.kp * error + self.kd * delerror_delt + self.ki * sum_error

    # def update_controller(self, learning_rate, delsumerror_delkp, delsumerror_delkd, delsumerror_delki):
    #     # may be necessary to do the derivatives in this file. May then follow abstract controller
    #     self.kp -= learning_rate * delsumerror_delkp
    #     self.kd -= learning_rate * delsumerror_delkd
    #     self.ki -= learning_rate * delsumerror_delki
    def update_controller(self, learning_rate, delsumerror_delomega):
        self.kp -= learning_rate * delsumerror_delomega[0]
        self.ki -= learning_rate * delsumerror_delomega[1]
        self.kd -= learning_rate * delsumerror_delomega[2]

    def reset_error_history(self):
        self.error_history = jnp.array([0])