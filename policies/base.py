from abc import ABC, abstractmethod

class BasePolicy(ABC):
    @abstractmethod
    def select_action(self, q_table, state):
        pass