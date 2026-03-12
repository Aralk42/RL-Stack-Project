import numpy as np
from tqdm import tqdm

""" from utils.metrics import moving_average, save_rewards
from utils.storage import save_q_table """

def train(env_class, agent_class, policy_class, n_episodes=500, max_steps=100, seed=None):

    env = env_class(seed=seed)
    policy = policy_class
    agent = agent_class(env.n_states, env.n_actions, policy=policy, seed=seed)
 
    #state_size = env.observation_space.nvec
    #action_size = env.action_space.n
    # agent = QLearningAgent(state_size, action_size)

    rewards = []

    for episode in tqdm(range(n_episodes)):

        state, _ = env.reset()

        done = False
        total_reward = 0

        for t in range(max_steps):

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
           # print(f"Episode {episode} Reward {total_reward}")

    return rewards, agent
