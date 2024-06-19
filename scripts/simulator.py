import random

from agent.e_greedy_agent import EGreedyAgent
from gym_el_farol.envs import ElFarolEnv, FuzzyPureNash
from agent.erev_roth_agent import ErevRothAgent


def iterate(agents, env):
    actions = [a.act() for a in agents]
    obs, reward, _, _ = env.step(actions)
    for agent, reward in zip(agents, reward):
        agent.learn(reward)
    return actions


def modify_threshold():
    if random.random() < threshold_change_chance:
        change = (random.random() - 0.5) * threshold_change_limit
        env.modify_threshold(change)


def iterations_to_equilibrium(agents, env):
    nash = FuzzyPureNash()
    for iter in range(0, 5000000):
        if iter % 50 == 0 and iter > 0:
            modify_threshold()
            if nash.in_equilibria():
                return iter
            nash = FuzzyPureNash()
        actions = iterate(agents, env)
        nash.step(actions)
    return False


threshold_change_chance = 0.3
threshold_change_limit = 0.3
n_agents = 100
env = ElFarolEnv(n_agents=n_agents, threshold=70)
agents = []
print("attended, threshold")
for i in range(0, n_agents):
    agents.append(EGreedyAgent(env.observation_space, env.action_space))
print(iterations_to_equilibrium(agents, env))
