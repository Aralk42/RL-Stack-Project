import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.OneSensorAndAntenna import WPT_1to1
from agents.q_learning import QLearningAgent
from policies.epsilon_greedy import EpsilonGreedyPolicy
from train import train
import datetime

def main():
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    SEED = 42
    # NumPy
    np.random.seed(SEED)


    # --- Configuración del experimento ---
    ENV = WPT_1to1          # Clase del environment
    AGENT = QLearningAgent   # Clase del agente
    POLICY = EpsilonGreedyPolicy  # Clase de la política

    N_EPISODES = 1000
    MAX_STEPS = 1200

    config = {
        "run_id":run_id,
        "algorithm": POLICY.__class__.__name__,
        "env": ENV.__class__.__name__,
        "agent": AGENT.__class__.__name__,
        "n_episodes": N_EPISODES,
        "max_steps": MAX_STEPS,
        "alpha": 0.1, #todo: change
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay":0.995,
        "epsilon_min": 0.1,
        "seed": SEED
    }

    # Llamada al entrenamiento
    rewards, agent = train(config = config, env_class=ENV, agent_class=AGENT, policy_class=POLICY)
    
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward")
    plt.show()
    
if __name__ == "__main__":
    main() 