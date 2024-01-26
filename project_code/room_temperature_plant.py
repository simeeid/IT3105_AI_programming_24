#from abstract_plant import AbstractPlant

import numpy as np
import jax
import jax.numpy as jnp

class RoomTemperaturePlant():#AbstractPlant):
    def __init__(self, temperature_outside, heat_transfer_coefficient, volume):
        self.temperature_outside = temperature_outside + 273.15 # K
        self.heat_transfer_coefficient = heat_transfer_coefficient
        self.volume = volume
        self.density = 1.2 # kg/m^3
        self.specific_heat_capacity = 1005 # J/(kg*K)

    def update_plant(self, control_signal, external_disturbance, temperature_inside_arr):
        temperature_inside = temperature_inside_arr[0]
        temperature_inside = temperature_inside + (control_signal / (self.density * self.specific_heat_capacity * self.volume)) * (self.temperature_outside - temperature_inside) + control_signal / (self.density * self.specific_heat_capacity * self.volume) + external_disturbance
        return [temperature_inside]
