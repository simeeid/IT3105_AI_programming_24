from abstract_plant import AbstractPlant

import numpy as np
import jax
import jax.numpy as jnp

class CournotCompetitionPlant():#AbstractPlant):
    def __init__(self, max_price, marginal_cost):
        self.max_price = max_price
        self.marginal_cost = marginal_cost

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

    def reset(self, initial_value):
        self.height_bathtub_water = initial_value