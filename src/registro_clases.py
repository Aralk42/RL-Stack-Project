import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.OneSensorAndAntenna import WPT_1to1
from agents.q_learning import QLearningAgent
from policies.epsilon_greedy import EpsilonGreedyPolicy

class RegistroClases():
    def __init__(self):
        pass
    def get_envs(self):
        return  {
            "WPT_1to1": WPT_1to1,
        }
    def get_agents(self):
        return {
            "QLearningAgent": QLearningAgent,
        }
    def get_policies(self):
        return {
            "EpsilonGreedyPolicy": EpsilonGreedyPolicy,
        }
    def get_component_names(self):
        names_envs = list(self.get_envs().keys())
        names_agents = list(self.get_agents().keys())
        names_policies = list(self.get_policies().keys())
        return names_envs, names_agents, names_policies


    def get_rl_components(self,config):
        name_env = config["env"]
        name_agent = config["agent"]
        name_policy = config["algorithm"]
        print(name_env, name_agent, name_policy)
        environments = self.get_envs()
        agents = self.get_agents()
        policies = self.get_policies()
        print(environments, agents, policies)
        return environments[str(name_env)], agents[str(name_agent)], policies[str(name_policy)]