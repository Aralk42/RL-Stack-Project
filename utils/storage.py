import numpy as np
import json
from pathlib import Path

def log_run(config):
    # Se llama primero para crear la carpeta y la configuración de toda la run.
    # todo: Asegurarse de que se llama primero esta antes de guardar las métricas y la qtable.
    
    results_dir = Path("experiments/results")
    runs_dir = results_dir / "runs"
    run_dir = runs_dir / config["run_id"]

    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    return run_dir

def log_metrics(run_dir,data):
    filepath = run_dir / "metrics.jsonl"

    with open(filepath, "a") as f:
        f.write(json.dumps(data) + "\n")

def save_q_table(run_dir, q_table):
    try: 
        np.save(run_dir / "qtable.npy", q_table)
    except: 
        print("An exception occurred whyle saving q table.")

def save_learning_curve(run_dir, rewards):
    try: 
        np.save(run_dir / "learning_curve.npy", np.array(rewards))
    except: 
        print("An exception occurred whyle saving the rewards.")