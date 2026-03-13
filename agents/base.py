from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def choose_action(self, state):
        pass

    @abstractmethod
    def update(self, state, action, reward, next_state):
        pass

    @abstractmethod
    def decay_epsilon(self):
        pass