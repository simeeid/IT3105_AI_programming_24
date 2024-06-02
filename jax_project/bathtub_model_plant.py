from config_reader import ConfigReader

import numpy as np
import jax
import jax.numpy as jnp

class BathtubModelPlant():
    """
    A class representing the bathtub model plant.

    Attributes:
        cross_sectional_area_bathtub (float): The cross-sectional area of the bathtub.
        cross_sectional_area_drain (float): The cross-sectional area of the drain.
        height_bathtub_water (float): The initial height of water in the bathtub.
        range_disturbance (float): The range of random disturbance for an epoch.

    Methods:
        update_plant(control_signal, external_disturbance, initial_value_arr):
            Updates the water level in the bathtub based on the control signal, external disturbance, and initial value.
        get_external_disturbance(size):
            Generates random disturbance values for an epoch.
        get_initial_value():
            Returns the specified initial value for the plant.
    """

    def __init__(self):
        config_reader = ConfigReader("project_code/config.json")

        self.cross_sectional_area_bathtub = config_reader.get_chosen_plant_config('bathtub_model')['cross_sectional_area_bathtub']
        self.cross_sectional_area_drain = config_reader.get_chosen_plant_config('bathtub_model')['cross_sectional_area_drain']
        self.height_bathtub_water = config_reader.get_chosen_plant_config('bathtub_model')['height_bathtub_water']

        self.range_disturbance = config_reader.get_consys_config()['range_disturbance']

    def update_plant(self, control_signal, external_disturbance, initial_value_arr):
        """
        Updates the water level in the bathtub based on the control signal, external disturbance, and initial value.

        Args:
            control_signal (float): The control signal from the controller for the plant.
            external_disturbance (float): The external disturbance for the plant.
            initial_value_arr (list): The initial value of the water level in the bathtub.

        Returns:
            list: The updated water level in the bathtub.
        """
        initial_value = initial_value_arr[0] # the initial value is the water level in the bathtub
        V = jnp.sqrt(2 * 9.81 * initial_value)
        Q = V * self.cross_sectional_area_drain
        delb_delt = control_signal + external_disturbance - Q
        delh_delt = delb_delt / self.cross_sectional_area_bathtub

        initial_value += delh_delt
        return [initial_value]

    def get_external_disturbance(self, size):
        """
        Generates random disturbance values for an epoch.

        Args:
            size (int): The size of the disturbance list to generate.

        Returns:
            ndarray: An array of random disturbance values.
        """
        return np.random.uniform(-self.range_disturbance, self.range_disturbance, size)
    
    def get_initial_value(self):
        """
        Returns the specified initial value for the plant.

        Returns:
            list: The specified initial value for the plant.
        """
        return [self.height_bathtub_water]