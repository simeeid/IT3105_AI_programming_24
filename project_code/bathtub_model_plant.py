#from abstract_plant import AbstractPlant

import numpy as np
import jax
import jax.numpy as jnp

class BathtubModelPlant():
    def __init__(self, cross_sectional_area_bathtub, cross_sectional_area_drain, height_bathtub_water):
        self.cross_sectional_area_bathtub = cross_sectional_area_bathtub
        self.cross_sectional_area_drain = cross_sectional_area_drain
        self.height_bathtub_water = height_bathtub_water

    def update_plant(self, control_signal, external_disturbance, height_bathtub_water):
        V = jnp.sqrt(2 * 9.81 * height_bathtub_water)
        Q = V * self.cross_sectional_area_drain
        delb_delt = control_signal + external_disturbance - Q
        delh_delt = delb_delt / self.cross_sectional_area_bathtub

        height_bathtub_water += delh_delt
        return height_bathtub_water

    def reset(self, height_bathtub_water):
        self.height_bathtub_water = height_bathtub_water