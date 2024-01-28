from abstract_plant import AbstractPlant
from config_reader import ConfigReader

import numpy as np
import jax
import jax.numpy as jnp

class BathtubModelPlant():
    def __init__(self):
        config_reader = ConfigReader("project_code/config.json")

        self.cross_sectional_area_bathtub = config_reader.get_chosen_plant_config('bathtub_model')['cross_sectional_area_bathtub']
        self.cross_sectional_area_drain = config_reader.get_chosen_plant_config('bathtub_model')['cross_sectional_area_drain']
        self.height_bathtub_water = config_reader.get_chosen_plant_config('bathtub_model')['height_bathtub_water']

        self.range_disturbance = config_reader.get_consys_config()['range_disturbance']

    def update_plant(self, control_signal, external_disturbance, initial_value_arr):
        initial_value = initial_value_arr[0]
        V = jnp.sqrt(2 * 9.81 * initial_value)
        Q = V * self.cross_sectional_area_drain
        delb_delt = control_signal + external_disturbance - Q
        delh_delt = delb_delt / self.cross_sectional_area_bathtub

        initial_value += delh_delt
        return [initial_value]

    def get_external_disturbance(self, size):
        return np.random.uniform(-self.range_disturbance, self.range_disturbance, size)
    
    def get_initial_value(self):
        return [self.height_bathtub_water]