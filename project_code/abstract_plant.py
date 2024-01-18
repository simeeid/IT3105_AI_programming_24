from abc import ABC, abstractmethod

class AbstractPlant(ABC):
    @abstractmethod
    def update_plant(self, control_signal, external_disturbance):
        pass
