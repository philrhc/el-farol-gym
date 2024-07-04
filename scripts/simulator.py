import random

from agent.e_greedy_agent import EGreedyAgent
from agent.erev_roth_agent import ErevRothAgent
from environment import ElFarolEnv, FuzzyPureNash


def iterate(agents, env):
    actions = [a.act() for a in agents]
    obs, reward, _, _ = env.step(actions)
    for agent, reward in zip(agents, reward):
        agent.learn(reward)
    return actions


def modify_capacity():
    if random.random() < capacity_change_chance:
        change = (random.random() - 0.5) * capacity_change_limit
        env.modify_capacity(change)


def iterations_to_equilibrium(agents, env, iterations, to_nash=False):
    nash = FuzzyPureNash()
    for iter in range(0, iterations):
        if iter % 50 == 0 and iter > 0:
            modify_capacity()
            if to_nash and nash.in_equilibria():
                return iter
            nash = FuzzyPureNash()
        actions = iterate(agents, env)
        nash.step(actions)
    return False


capacity_change_chance = 0.3
capacity_change_limit = 0.2
iterations = 5000
n_agents = 100
env = ElFarolEnv(n_agents=n_agents)
agents = []


def main():
    for i in range(0, n_agents):
        agents.append(EGreedyAgent(env.action_space))
    iterations_to_equilibrium(agents, env, iterations, False)
    env.plot_attendance_and_capacity(iterations)

if __name__ == '__main__':
    main()
