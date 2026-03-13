import numpy as np
from tqdm import tqdm
from utils.metrics import moving_average
from utils.storage import log_run, save_q_table, log_metrics, save_learning_curve

""" from utils.metrics import moving_average, save_rewards
from utils.storage import save_q_table """

def train(config, env_class, agent_class, policy_class):

    env = env_class(seed=config["seed"])
    policy = policy_class
    agent = agent_class(env.n_states, env.n_actions, policy=policy, seed=config["seed"])
 
    #state_size = env.observation_space.nvec
    #action_size = env.action_space.n
    # agent = QLearningAgent(state_size, action_size)

    run_dir = log_run(config)

    rewards = []

    for episode in tqdm(range(config["n_episodes"])):

        state, _ = env.reset()

        done = False
        total_reward = 0

        for t in range(config["max_steps"]):

            action = agent.choose_action(state)

            next_state, reward, done, _ = env.step(action)

            agent.update(state, action, reward, next_state)

            state = next_state
            total_reward += reward

            if done:
                break


        agent.decay_epsilon()

        rewards.append(total_reward)

        #if episode % 100 == 0:
           # moving_avg = moving_average(rewards, window=100)

        log_metrics(run_dir, {
            "episode": episode,
            "reward": total_reward,
            #"moving_avg": moving_avg
        })
    
    # Guardar resultados y modelo
    save_q_table(run_dir,agent.q_table)
    save_learning_curve(run_dir, rewards)
    return rewards, agent
