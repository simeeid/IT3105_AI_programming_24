from config_reader import ConfigReader

import numpy as np
import jax
import jax.numpy as jnp

class CournotCompetitionPlant():
    def __init__(self):
        """
        Initializes the CournotCompetitionPlant class.

        Reads the configuration from the 'config.json' file and sets the maximum price,
        marginal cost, range of disturbance, initial values for q1 and q2, and target value.
        """
        config_reader = ConfigReader("project_code/config.json")

        self.max_price = config_reader.get_chosen_plant_config('cournot_model')['max_price']
        self.marginal_cost = config_reader.get_chosen_plant_config('cournot_model')['marginal_cost']

        self.range_disturbance = config_reader.get_consys_config()['range_disturbance']
        self.q1_initial_value = config_reader.get_chosen_plant_config('cournot_model')['q1_initial_value']
        self.q2_initial_value = config_reader.get_chosen_plant_config('cournot_model')['q2_initial_value']
        self.target_value = config_reader.get_chosen_plant_config('cournot_model')['target_value']

    def update_plant(self, control_signal, external_disturbance, value_arr):
        """
        Updates the profit made by competitor 1 at each time step.

        Args:
            control_signal (float): The control signal for competitor 1.
            external_disturbance (float): The external disturbance for competitor 2.
            value_arr (list): The array of values [t, q1, q2].

        Returns:
            tuple: A tuple containing the profit made by competitor 1, the updated q1 value,
                   and the updated q2 value.
        """
        def p(q): return self.max_price - q

        q1_initial_value = value_arr[1]
        q2_initial_value = value_arr[2]

        # make sure that q1_t_plus_1 and q2_t_plus_1 are 0<=x<=1
        q1_t_plus_1 = jnp.clip(control_signal + q1_initial_value, 0, 1)
        q2_t_plus_1 = jnp.clip(external_disturbance + q2_initial_value, 0, 1)

        # q is the total quantity produced by both competitors
        q = q1_t_plus_1 + q2_t_plus_1
        price = p(q)

        profit_1 = q1_t_plus_1 * (price - self.marginal_cost)
        return profit_1, q1_t_plus_1, q2_t_plus_1

    def get_external_disturbance(self, size):
        """
        Generates a random disturbance for an epoch using this plant.

        Args:
            size (int): The size of the disturbance array.

        Returns:
            numpy.ndarray: An array of random disturbances within the specified range.
        """
        return np.random.uniform(-self.range_disturbance, self.range_disturbance, size)
    
    def get_initial_value(self):
        """
        Returns the specified initial value for the plant.

        Returns:
            list: A list containing the target value, q1 initial value, and q2 initial value.
        """
        return [self.target_value, self.q1_initial_value, self.q2_initial_value]