import numpy as np
from policies import policiy

class QLearningAgent: 
    def __init__(self, state_size, action_size,
                 learning_rate=0.1,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_decay=0.995,
                 epsilon_min=0.1):
        self.state_size = np.array(state_size)
        # print(self.state_size)
        self.action_size = action_size
        self.lr = learning_rate
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = np.zeros((*state_size, action_size)) #todo ??

        self.policy = policiy().epsilon_greedy_policy

    def choose_action(self, state):
        action = self.policy(self.q_table, state,self.epsilon)
        return action
    
    def update(self, state, action, reward, next_state):

        best_next_action = np.max(self.q_table[next_state[0],next_state[1]])

        td_target = reward + self.gamma * best_next_action

        td_error = td_target - self.q_table[state[0],state[1]][action]

        self.q_table[state[0],state[1]][action] += self.lr * td_error

    def decay_epsilon(self):

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay