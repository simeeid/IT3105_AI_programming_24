from abc import ABC, abstractmethod

class AbstractPlant(ABC):
    @abstractmethod
    def update_plant(self, control_signal, external_disturbance, initial_value):
        pass

    @abstractmethod
    def reset(self, initial_value):
        pass
