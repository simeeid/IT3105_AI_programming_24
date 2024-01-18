from abc import ABC, abstractmethod

class AbstractController(ABC):
    @abstractmethod
    def compute_control_signal(self, error, delerror_delt, sum_error):
        pass

    @abstractmethod
    def update_controller(self, error, delerror_delt, sumerror):
        pass