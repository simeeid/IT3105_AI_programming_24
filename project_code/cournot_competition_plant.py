from config_reader import ConfigReader

import numpy as np
import jax
import jax.numpy as jnp

class CournotCompetitionPlant():
    def __init__(self):
        config_reader = ConfigReader("project_code/config.json")

        self.max_price = config_reader.get_chosen_plant_config('cournot_model')['max_price']
        self.marginal_cost = config_reader.get_chosen_plant_config('cournot_model')['marginal_cost']

        self.range_disturbance = config_reader.get_consys_config()['range_disturbance']
        self.q1_initial_value = config_reader.get_chosen_plant_config('cournot_model')['q1_initial_value']
        self.q2_initial_value = config_reader.get_chosen_plant_config('cournot_model')['q2_initial_value']
        self.target_value = config_reader.get_chosen_plant_config('cournot_model')['target_value']

    def update_plant(self, control_signal, external_disturbance, value_arr):
        def p(q): return self.max_price - q

        q1_initial_value = value_arr[1]
        q2_initial_value = value_arr[2]

        # make sure that q1_t_plus_1 and q2_t_plus_1 are 0<=x<=1
        q1_t_plus_1 = jnp.clip(control_signal + q1_initial_value, 0, 1)
        q2_t_plus_1 = jnp.clip(external_disturbance + q2_initial_value, 0, 1)

        q = q1_t_plus_1 + q2_t_plus_1
        price = p(q)

        profit_1 = q1_t_plus_1 * (price - self.marginal_cost)
        return profit_1, q1_t_plus_1, q2_t_plus_1

    def get_external_disturbance(self, size):
        return np.random.uniform(-self.range_disturbance, self.range_disturbance, size)
    
    def get_initial_value(self):
        return [self.target_value, self.q1_initial_value, self.q2_initial_value]