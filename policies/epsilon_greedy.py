import numpy as np
from .base import BasePolicy

class EpsilonGreedyPolicy(BasePolicy):
    def __init__(self, epsilon=0.1,seed=None):
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def select_action(self,q_table,state):  # Input: q for the given state
        
        if self.rng.random(1) < self.epsilon:
            action = self.rng.choice(np.arange(q_table.shape[-1])) 
            # Return an action uniformly
        else:
            action = np.argmax(q_table[*state])  
            # Action that maximizes q
        return int(action)