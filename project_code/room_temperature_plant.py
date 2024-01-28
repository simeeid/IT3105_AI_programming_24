from config_reader import ConfigReader

import numpy as np
import jax
import jax.numpy as jnp

class RoomTemperaturePlant():
    def __init__(self):
        config_reader = ConfigReader("project_code/config.json")

        self.target_temperature = config_reader.get_chosen_plant_config('room_model')['target_temperature']
        self.temperature_outside = config_reader.get_chosen_plant_config('room_model')['temperature_outside']
        self.volume = config_reader.get_chosen_plant_config('room_model')['volume']
        self.surface_area = config_reader.get_chosen_plant_config('room_model')['surface_area']
        self.thermal_conductivity = config_reader.get_chosen_plant_config('room_model')['thermal_conductivity']
        self.wall_thickness = config_reader.get_chosen_plant_config('room_model')['wall_thickness']

        self.temperature_outside = self.temperature_outside + 273.15 # K
        self.density = 1.2 # kg/m^3
        self.specific_heat_capacity = 1005 # J/(kg*K)

    def update_plant(self, control_signal, external_disturbance, temperature_inside_arr):
        temperature_inside = temperature_inside_arr[0] + 273.15 # K
        thermal_capacitance = self.volume * self.density * self.specific_heat_capacity
        thermal_resistance = self.wall_thickness / (self.thermal_conductivity * self.surface_area)
        temperature_inside = temperature_inside + (self.temperature_outside - temperature_inside) / (thermal_resistance * thermal_capacitance) + 50 * control_signal / thermal_capacitance + external_disturbance
        return [temperature_inside - 273.15]

    def get_external_disturbance(self, size):
        external_disturbance = -np.random.normal(0, 0.01, size)
        external_disturbance = external_disturbance - 0.02
        external_disturbance = np.clip(external_disturbance, -0.025, 0)
        return external_disturbance
    
    def get_initial_value(self):
        return [self.target_temperature]