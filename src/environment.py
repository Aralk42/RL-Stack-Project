import gymnasium as gs
import numpy as np

class WPT_1to1(object):
    def __init__(self, battery_units = 12,seed=42):
        self.bu = battery_units

        # Los estados del sistema son los niveles de batería del sensor
        self.observation_space = gs.Discrete(battery_units)
        # Acciones del sistema: nada, enviar energía.
        self.action_space = gs.MultiBinary(1, seed=seed) #0, no hace nada, 1, envía potencia.
        self.init_state = battery_units - 1  # El sensor inicia con un nivel de batería de 11 unidades
        self.state = None  # This is to store data later on

    def reset(self, randomize=False): 
        if randomize:
            self.state = self.observation_space.sample()
        else:
            self.state = self.init_state
        return self.state, {}
    
    def step(self, action):

        action = np.squeeze(action).astype(int).item()
        if action != 0 and action != 1:  # Check the action bounds
            raise RuntimeError('Action out of bounds')
        if action == 1:
            next_state = self.state + 1