import numpy as np

from environment import WPT_1to1
from agent import QLearningAgent

def train(episodes=2000, total_time = 1440):

    env = WPT_1to1()
 
    state_size = env.observation_space.nvec
    action_size = env.action_space.n
    agent = QLearningAgent(state_size, action_size)

    rewards = []

    for episode in range(episodes):

        state, _ = env.reset()

        done = False
        total_reward = 0

        for t in range(total_time):

            action = agent.choose_action(state)

            next_state, reward, done, _ = env.step(action)

            if done:
                break

            agent.update(state, action, reward, next_state)

            state = next_state

            total_reward += reward

        agent.decay_epsilon()

        rewards.append(total_reward)

        if episode % 100 == 0:
            print(f"Episode {episode} Reward {total_reward}")

    return rewards, agent
