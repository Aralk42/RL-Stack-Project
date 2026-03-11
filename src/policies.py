import numpy as np


class policiy:
    def __init__(self,rng=42):
        self.rng = np.random.default_rng(rng)

    def epsilon_greedy_policy(self,q_table,state,epsilon):  # Input: q for the given state
        if self.rng.random(1) < epsilon:
            action = self.rng.choice(np.arange(q_table.shape[2])) 
            # Return an action uniformly
        else:
            action = np.argmax(q_table[state[0],state[1]])  
            # Action that maximizes q
        return int(action)