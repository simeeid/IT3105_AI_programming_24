import json

class ConfigReader:
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path
        self.config_data = self._load_config()

    def _load_config(self):
        with open(self.config_file_path, 'r') as file:
            return json.load(file)
        
    def get_plant_config(self):
        return self.config_data['plant']
      
    def get_chosen_plant_config(self, plant_name):
        return self.config_data[plant_name]

    def get_controller_config(self):
        return self.config_data['controller']
    
    def get_neural_net_config(self):
        return self.config_data['neural_net']
    
    def get_consys_config(self):
        return self.config_data['consys']