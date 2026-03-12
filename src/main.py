import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.OneSensorAndAntenna import WPT_1to1
from agents.q_learning import QLearningAgent
from policies.epsilon_greedy import EpsilonGreedyPolicy
from train import train

def main():
    SEED = 42
    # NumPy
    np.random.seed(SEED)


    # --- Configuración del experimento ---
    ENV = WPT_1to1          # Clase del environment
    AGENT = QLearningAgent   # Clase del agente
    POLICY = EpsilonGreedyPolicy  # Clase de la política

    N_EPISODES = 1000
    MAX_STEPS = 1200

    # Llamada al entrenamiento
    rewards, agent = train(env_class=ENV, agent_class=AGENT, policy_class=POLICY,
        n_episodes=N_EPISODES, max_steps=MAX_STEPS, seed= SEED)
    
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward")
    plt.show()
    
if __name__ == "__main__":
    main() 