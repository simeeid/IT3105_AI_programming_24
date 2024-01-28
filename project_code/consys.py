from matplotlib import pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from classic_pid_controller import ClassicPidController
from neural_net_controller import NeuralNetController
from bathtub_model_plant import BathtubModelPlant
from cournot_competition_plant import CournotCompetitionPlant
from room_temperature_plant import RoomTemperaturePlant

from config_reader import ConfigReader
config_reader = ConfigReader("project_code/config.json")

# ==================== initialize system ====================
def initialize_system():
    global chosen_controller; global plant
    global num_epochs; global num_timesteps; global learning_rate; global range_disturbance
    global range_k_values
    global num_layers; global num_neurons; global activation_function; global range_initial_value

    num_epochs = config_reader.get_consys_config()['num_epochs']
    num_timesteps = config_reader.get_consys_config()['num_timesteps']
    learning_rate = config_reader.get_consys_config()['learning_rate']
    range_disturbance = config_reader.get_consys_config()['range_disturbance']

    chosen_plant = config_reader.get_plant_config()['value']
    if chosen_plant == "bathtub_model":
        plant = BathtubModelPlant()
    elif chosen_plant == "cournot_model":
        plant = CournotCompetitionPlant()
    elif chosen_plant == "room_model":
        plant = RoomTemperaturePlant()
    else:
        raise ValueError(f"Plant \"{chosen_plant}\" not found")

    chosen_controller = config_reader.get_controller_config()['value']
    if chosen_controller == "neural_net":
        num_layers = config_reader.get_chosen_controller_config('neural_net')['num_layers']
        num_layers += 2
        num_neurons = config_reader.get_chosen_controller_config('neural_net')['num_neurons']
        num_neurons = [3] + num_neurons + [1]
        activation_function = config_reader.get_chosen_controller_config('neural_net')['activation']
        range_initial_value = config_reader.get_chosen_controller_config('neural_net')['range_initial_value']
    
        if len(num_neurons) != num_layers:
            raise ValueError(f"Length of num_neurons ({len(num_neurons) - 2}) is unequal to num_layers ({num_layers - 2})")
    elif chosen_controller == "classic":
        range_k_values = config_reader.get_chosen_controller_config('classic')['range_k_values']
    else:
        raise ValueError(f"Controller \"{chosen_controller}\" not found")

def initialize_weights_and_biases():
    if chosen_controller == "neural_net":
        layers = num_neurons
        sender = layers[0]; params = []
        for reciever in layers[1:]:
            weights = np.random.uniform(-range_initial_value,range_initial_value,(sender,reciever))
            biases = np.random.uniform(-range_initial_value,range_initial_value,(1,reciever))
            sender = reciever
            params.append((weights,biases))
    elif chosen_controller == "classic":
        params = np.random.uniform(0, range_k_values, size=(3, ))
    return params

# ==================== run system ====================
def compute_mse(error):
    return jnp.mean(jnp.square(error))

def run_epoch(params):
    if chosen_controller == "neural_net":
        controller = NeuralNetController(params, num_layers, num_neurons, activation_function)
        control_signal = controller.compute_control_signal(params, jnp.array([0.0, 0.0, 0.0]), activation_function)
    elif chosen_controller == "classic":
        controller = ClassicPidController(params)
        control_signal = controller.compute_control_signal(0, 0, 0)

    external_disturbance = plant.get_external_disturbance(num_timesteps)
    local_value_arr = plant.get_initial_value()
    target_value = local_value_arr[0]

    for i in range(num_timesteps):
        local_value_arr = plant.update_plant(control_signal, external_disturbance[i], local_value_arr)
        error = target_value - local_value_arr[0]
        controller.update_error_history(error)
        delerror_delt = controller.error_history[-1] - controller.error_history[-2]

        if chosen_controller == "neural_net":
            control_signal = controller.compute_control_signal(params, jnp.array([error[0][0], delerror_delt, jnp.sum(controller.error_history)]), activation_function)
        elif chosen_controller == "classic":
            control_signal = controller.compute_control_signal(error, delerror_delt, jnp.sum(controller.error_history))

    mse = compute_mse(controller.error_history)
    return mse
        
def update_controller(params, gradient):
    if chosen_controller == "neural_net":
        num = 0
        for weights, biases in params:
            for i in range(len(weights)):
                weights[i] -= learning_rate * gradient[num][0][i]
            for i in range(len(biases)):
                biases[i] -= learning_rate * gradient[num][1][i]
            num += 1
    elif chosen_controller == "classic":
        params -= learning_rate * gradient
    return params

def plot_run(mse_history, params_history):
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("MSE history")
    plt.plot(mse_history)
    plt.show()
    if chosen_controller == "classic":
        params_history = np.array(params_history)
        plt.xlabel("Epoch")
        plt.ylabel("Y")
        plt.title("PID controller parameters history")
        plt.plot(params_history[:, 0], label="kp")
        plt.plot(params_history[:, 1], label="ki")
        plt.plot(params_history[:, 2], label="kd")
        plt.legend()
        plt.show()

def run_m_epoch(m):
    mse_history = []
    params_history = []
    params = initialize_weights_and_biases()

    delsumerror_delomega1 = jax.value_and_grad(run_epoch, argnums=0)
    delsumerror_delomega = jax.jit(delsumerror_delomega1)
    for _ in range(m):
        mse, gradient = delsumerror_delomega(params)
        if chosen_controller == "classic":
            params_history.append(params)
        params = update_controller(params, gradient)
        mse_history.append(mse)
    plot_run(mse_history, params_history)

if __name__ == "__main__":
    initialize_system()
    run_m_epoch(num_epochs)