import gymnasium as gym
import gymnasium.spaces as gs
import numpy as np

class WPT_1to1(gym.Env):
    def __init__(self, battery_units = 12,seed=42):

        super().__init__() #todo ??

        self.max_battery = battery_units
        # Estado = (battery_level, time_mod_60)
        self.observation_space = gs.MultiDiscrete(
            [self.max_battery +1,60], 
            seed=seed
            )
        
        # Acciones: 0 no enviar, 1 enviar
        self.action_space = gs.Discrete(2, seed=seed)
      
        # Parámetros del sistema
        self.send_data_time = 60
        self.measure_gap = 5

        self.battery_wasted_on_sending_data = 3
        self.battery_wasted_on_measuring_data = 1

        self.energy_sent_per_second = 1

        # Estado inicial
        self.init_battery = battery_units - 1
        self.state = None  # This is to store data later on
        self.time = 0

    # Reset the environment
    def reset(self, seed=None): 
        super().reset(seed=seed)
        self.time = 0
        battery = self.init_battery
        self.state = np.array([battery, self.time % 60], dtype=int)
        return self.state, {}


    # Environment transition
    def step(self, action):

        current_battery, _ = self.state

        action = int(action)

        self.time += 1
        time_mod = self.time % 60

        energy_sent = 0
        if action == 1:
            energy_sent = self.energy_sent_per_second

            
        next_battery = WPT_1to1.sensor_behaviour(self,energy_sent,current_battery)
        # limitar batería
        next_battery = np.clip(next_battery, 0, self.max_battery)

        next_state = np.array([next_battery, time_mod], dtype=int)
        
        reward, done = self.reward_done(next_battery, action)

        self.state = next_state

        return next_state, reward, done, {} 
    
    
    def reward_done(self, battery, action): 

        reward = 0 
        # reward += battery / self.max_battery #todo
        terminated = False
        collecting_energy = not (
            self.time % 60 == 0 or
            self.time % 5 == 0
        )
        # batería agotada
        if battery == 0:
            reward = -100
            terminated = True
        # penalizar envío inútil
        if action == 1 and not collecting_energy:
            reward -= 2

        # penalizar batería muy baja
        if battery < 3:
            reward -= 1

        # pequeña penalización por enviar energía
        if action == 1:
            reward -= 0.1
            
        return reward, terminated
    
    # todo: La información del sensor no estaría disponible hasta que el sensor enviara datos.
    # todo: 
    
    def sensor_behaviour(self,energy_sent,current_battery):
        battery = current_battery

        if self.time % self.send_data_time == 0:
            battery -= self.battery_wasted_on_sending_data
        elif self.time % self.measure_gap == 0:
            battery -= self.battery_wasted_on_measuring_data
        else:
            battery += energy_sent

        return battery

