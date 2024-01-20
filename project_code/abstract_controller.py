from abc import ABC, abstractmethod

class AbstractController(ABC):
    @abstractmethod
    def update_error_history(self, error):
        pass

    @abstractmethod
    def compute_control_signal(self, params):
        pass