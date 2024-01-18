import numpy as np
import jax
import jax.numpy as jnp

from classic_pid_controller import ClassicPidController
# from test import NeuralNetController
from bathtub_model_plant import BathtubModelPlant
# from config_reader import ConfigReader

# learning_rate = 0.1
# water_height = 1

# plant = BathtubModelPlant(10, 0.1, water_height)
# #controller = ClassicPidController(1, 1, 1, 0.1)
# controller = NeuralNetController()

def compute_mse(error):
    return jnp.mean(jnp.square(error))

def show(mse):
    print(mse)

def compute_sum(error):
    return jnp.sum(error)

k = 10

# the following will be collected from the config file
cross_sectional_area_bathtub, cross_sectional_area_drain, height_bathtub_water = 25, 0.1, 10
learning_rate = 0.1

plant = BathtubModelPlant(cross_sectional_area_bathtub, cross_sectional_area_drain, height_bathtub_water)

#delerror_delt = jax.grad(controller.update_error_history, argnums=0)

def run_epoch(kp, ki, kd):
    controller = ClassicPidController(kp, ki, kd)

    local_height_bathtub_water = height_bathtub_water

    control_signal = controller.compute_control_signal(0, 0, 0)
    external_disturbance = jax.random.uniform(jax.random.PRNGKey(42), shape=(k,), minval=0.0, maxval=0.5)

    for i in range(k):
        plant_value = plant.update_plant(control_signal, external_disturbance[i], local_height_bathtub_water)
        local_height_bathtub_water = plant_value
        error = height_bathtub_water - plant_value
        controller.update_error_history(error)
        #derivative = delerror_delt(error) # either something like this, or the following line
        delerror_delt = controller.error_history[-1] - controller.error_history[-2]
        control_signal = controller.compute_control_signal(error, delerror_delt, jnp.sum(controller.error_history))

    # find mean squared error
    mse = compute_mse(controller.error_history)
    #error_sum = controller.error_history.sum()
    #print("error_sum: ", error_sum)
    #controller.reset_error_history()
    return mse
        
def run_m_epoch(m):
    kp, ki, kd = jax.random.uniform(jax.random.PRNGKey(42), shape=(3, ), minval=0.0, maxval=1.0)
    print("kp: ", kp, "ki: ", ki, "kd: ", kd)
    delsumerror_delomega = jax.value_and_grad(run_epoch, argnums=[0, 1, 2])
    #delsumerror_delomega = jax.jit(delsumerror_delomega)
    for _ in range(m):
        #run_epoch(kp, ki, kd)
        mse, gradient = delsumerror_delomega(kp, ki, kd)
        #print("kp: ", kp, "ki: ", ki, "kd: ", kd, "gradient: ", gradient)
        kp -= learning_rate * gradient[0]
        ki -= learning_rate * gradient[1]
        kd -= learning_rate * gradient[2]
        print("kp: ", kp, "ki: ", ki, "kd: ", kd, "mse: ", mse)
    #print("kp: ", kp, "ki: ", ki, "kd: ", kd)


run_m_epoch(10)
        




    # error = []
    # plant.reset(1)

    # external_disturbance = np.random.rand() * 0.04 + 0.01

    # for j in range(1):
    #     plant.update_plant(controller.compute_output(0, 0, 0), external_disturbance)
    #     # controller.update(learning_rate, 0, 0, 0)
    #     error.append(jnp.abs(plant.height_bathtub_water - water_height))

    # MSE = compute_mse(error)
    # delmse_delomega = jax.grad(MSE)([controller.kp, controller.ki, controller.kd])
    # controller.update(learning_rate, delmse_delomega[0], delmse_delomega[1], delmse_delomega[2])

    # make a jnp array with 3 floats
    