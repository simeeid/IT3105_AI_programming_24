from abstract_plant import AbstractPlant

import numpy as np
import jax
import jax.numpy as jnp

class BathtubModelPlant(AbstractPlant):
    def __init__(self, cross_sectional_area_bathtub, cross_sectional_area_drain, height_bathtub_water):
        self.cross_sectional_area_bathtub = cross_sectional_area_bathtub
        self.cross_sectional_area_drain = cross_sectional_area_drain
        self.height_bathtub_water = height_bathtub_water

    def update_plant(self, control_signal, external_disturbance, initial_value):
        V = jnp.sqrt(2 * 9.81 * initial_value)
        Q = V * self.cross_sectional_area_drain
        delb_delt = control_signal + external_disturbance - Q
        delh_delt = delb_delt / self.cross_sectional_area_bathtub

        initial_value += delh_delt
        return initial_value

    def reset(self, initial_value):
        self.height_bathtub_water = initial_value