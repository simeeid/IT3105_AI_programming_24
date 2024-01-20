import numpy as np
import jax
import jax.numpy as jnp

from classic_pid_controller import ClassicPidController
# from test import NeuralNetController
from abstract_controller import AbstractController
from bathtub_model_plant import BathtubModelPlant

from config_reader import ConfigReader
config_reader = ConfigReader("project_code/config.json")

# ==================== initialize system ====================
def initialize_system():
    global num_epochs; global num_timesteps; global learning_rate; global range_disturbance
    global cross_sectional_area_bathtub; global cross_sectional_area_drain; global height_bathtub_water; global range_k_values
    global plant

    num_epochs = config_reader.get_consys_config()['num_epochs']
    num_timesteps = config_reader.get_consys_config()['num_timesteps']
    learning_rate = config_reader.get_consys_config()['learning_rate']
    range_disturbance = config_reader.get_consys_config()['range_disturbance']

    cross_sectional_area_bathtub = config_reader.get_chosen_plant_config('bathtub_model')['cross_sectional_area_bathtub']
    cross_sectional_area_drain = config_reader.get_chosen_plant_config('bathtub_model')['cross_sectional_area_drain']
    height_bathtub_water = config_reader.get_chosen_plant_config('bathtub_model')['height_bathtub_water']
    range_k_values = config_reader.get_chosen_plant_config('bathtub_model')['range_k_values']

    plant = BathtubModelPlant(cross_sectional_area_bathtub, cross_sectional_area_drain, height_bathtub_water)

# ==================== run system ====================
def compute_mse(error):
    return jnp.mean(jnp.square(error))

def run_epoch(kp, ki, kd):
    controller = ClassicPidController(kp, ki, kd)

    local_height_bathtub_water = height_bathtub_water

    control_signal = controller.compute_control_signal(0, 0, 0)
    external_disturbance = jax.random.uniform(jax.random.PRNGKey(42), shape=(num_timesteps,), minval=0.0, maxval=range_disturbance)

    for i in range(num_timesteps):
        plant_value = plant.update_plant(control_signal, external_disturbance[i], local_height_bathtub_water)
        local_height_bathtub_water = plant_value
        error = height_bathtub_water - plant_value
        controller.update_error_history(error)
        delerror_delt = controller.error_history[-1] - controller.error_history[-2]
        control_signal = controller.compute_control_signal(error, delerror_delt, jnp.sum(controller.error_history))

    mse = compute_mse(controller.error_history)
    #controller.reset_error_history()
    return mse
        
def run_m_epoch(m):
    kp, ki, kd = jax.random.uniform(jax.random.PRNGKey(42), shape=(3, ), minval=0.0, maxval=range_k_values)
    print("kp: ", kp, "ki: ", ki, "kd: ", kd)
    delsumerror_delomega1 = jax.value_and_grad(run_epoch, argnums=[0, 1, 2])
    delsumerror_delomega = jax.jit(delsumerror_delomega1)
    for _ in range(m):
        mse, gradient = delsumerror_delomega(kp, ki, kd)
        kp -= learning_rate * gradient[0]
        ki -= learning_rate * gradient[1]
        kd -= learning_rate * gradient[2]
        print("kp: ", kp, "ki: ", ki, "kd: ", kd, "mse: ", mse) # to see development


if __name__ == "__main__":
    initialize_system()
    run_m_epoch(num_epochs)