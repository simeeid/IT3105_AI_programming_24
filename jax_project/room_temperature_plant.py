from config_reader import ConfigReader

import numpy as np
import jax
import jax.numpy as jnp

class RoomTemperaturePlant():
    def __init__(self):
        """
        Initializes the RoomTemperaturePlant object.

        Reads the configuration from the 'config.json' file and sets the initial values for target temperature,
        outside temperature, volume, surface area, thermal conductivity, and wall thickness.

        Converts the outside temperature to Kelvin and sets the density and specific heat capacity values.

        Parameters:
        None

        Returns:
        None
        """
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
        """
        Updates the plant's temperature based on the control signal, external disturbance, and current inside temperature.

        Calculates the thermal capacitance and thermal resistance of the room.
        Updates the inside temperature based on the temperature difference between inside and outside,
        control signal, external disturbance, and the thermal properties of the room.

        Parameters:
        control_signal (float): The control signal applied to the room.
        external_disturbance (float): The external disturbance affecting the room.
        temperature_inside_arr (list): The current inside temperature of the room.

        Returns:
        list: The updated inside temperature of the room.
        """
        temperature_inside = temperature_inside_arr[0] + 273.15 # K
        thermal_capacitance = self.volume * self.density * self.specific_heat_capacity
        thermal_resistance = self.wall_thickness / (self.thermal_conductivity * self.surface_area)
        temperature_inside = temperature_inside + (self.temperature_outside - temperature_inside) / (thermal_resistance * thermal_capacitance) + 50 * control_signal / thermal_capacitance + external_disturbance
        return [temperature_inside - 273.15]

    def get_external_disturbance(self, size):
        """
        Generates the external disturbance affecting the room.

        Generates random disturbance values and applies a range and offset to the values.

        Parameters:
        size (int): The size of the disturbance array.

        Returns:
        ndarray: The array of external disturbance values.
        """
        external_disturbance = -np.random.normal(0, 0.01, size)
        external_disturbance = external_disturbance - 0.02
        external_disturbance = np.clip(external_disturbance, -0.025, 0)
        return external_disturbance
    
    def get_initial_value(self):
        """
        Returns the initial value of the inside temperature.

        Parameters:
        None

        Returns:
        list: The initial value of the inside temperature.
        """
        return [self.target_temperature]