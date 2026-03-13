import numpy as np
import matplotlib.pyplot as plt
import json
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .registro_clases import RegistroClases
from .train import train
import datetime

def run_simulation(config_path):
    rc = RegistroClases()
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(config_path) as f:
        config = json.load(f)
    
    config["run_id"] = run_id
    np.random.seed(config["seed"])

    env_class, agent_class, policy_class = rc.get_rl_components(config)
    
    rewards, agent = train(
        config=config,
        env_class=env_class,
        agent_class=agent_class,
        policy_class=policy_class
    )