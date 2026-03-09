import numpy as np

from environment import create_environment
from agent import QLearningAgent


def train(episodes=10000):

    env = create_environment()

    # Spaces is a module to implement spaces in an environment: class gymnasium.spaces.Space(Box, Discrete...) 
    state_size = env.observation_space.n 
    # self.action_space = spaces.Discrete(3). Discrete: A space consisting of finitely many elements. 
    action_size = env.action_space.n # discrete.n: The number of elements of this space. en este caso (16,4)

    agent = QLearningAgent(state_size, action_size)

    rewards = []

    for episode in range(episodes):

        state, _ = env.reset()

        done = False
        total_reward = 0

        while not done:

            action = agent.choose_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            agent.update(state, action, reward, next_state)

            state = next_state

            total_reward += reward

        agent.decay_epsilon()

        rewards.append(total_reward)

        if episode % 100 == 0:
            print(f"Episode {episode} Reward {total_reward}")

    return rewards, agent