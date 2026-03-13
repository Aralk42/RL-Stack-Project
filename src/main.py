import numpy as np
import matplotlib.pyplot as plt
import json
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from registro_clases import RegistroClases
from train import train
import datetime

def main():
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    rc = RegistroClases()

    SEED = 42
    # NumPy
    np.random.seed(SEED)

    N_EPISODES = 1000
    MAX_STEPS = 1200

    config = {
        "run_id":run_id,
        "algorithm": "EpsilonGreedyPolicy",
        "env": "WPT_1to1",
        "agent": "QLearningAgent",
        "n_episodes": N_EPISODES,
        "max_steps": MAX_STEPS,
        "alpha": 0.1, #todo: change
        "gamma": 0.99,
        "epsilon": 1.0,
        "epsilon_decay":0.995,
        "epsilon_min": 0.1,
        "seed": SEED
    }
    env_class, agent_class, policy_class = rc.get_rl_components(config)

    # Llamada al entrenamiento
    rewards, agent = train(config = config, env_class=env_class, agent_class=agent_class, policy_class=policy_class)
    
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward")
    plt.show()

    
if __name__ == "__main__":
    main() 