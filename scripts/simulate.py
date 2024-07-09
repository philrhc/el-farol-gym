import random

from agent.e_greedy import EGreedyAgent
from agent.simple_erev_roth import SimpleErevRothAgent
from agent.erev_roth import ErevRothAgent
from environment import ElFarolEnv


def iterate(agents, env):
    actions = [a.act() for a in agents]
    obs, reward, _, _ = env.step(actions)
    for agent, reward in zip(agents, reward):
        agent.learn(reward)
    return actions


def random_threshold_changes(agents, env, iterations):
    for i in range(0, iterations):
        if i % 50 == 0 and i > 0:
            if random.random() < capacity_change_chance:
                change = (random.random() - 0.5) * capacity_change_limit
                env.modify_capacity_by_percentage(change)
        iterate(agents, env)


def static_threshold(agents, env, iterations):
    for i in range(0, iterations):
        iterate(agents, env)


def one_threshold_change(agents, env, iterations):
    for i in range(0, iterations):
        if i == int(iterations / 2):
            env.modify_capacity(50)
        iterate(agents, env)


iterations = 10_000
n_agents = 100
init_capacity = 70
capacity_change_chance = 0.2
capacity_change_limit = 0.3


def simulate(agents,
             env,
             visualise=False,
             simulation=random_threshold_changes):
    simulation(agents, env, iterations)
    if visualise:
        env.plot_attendance_and_capacity(iterations)
    return env.mse()


def simulate_erevroth(visualise=False,
                      simulation=static_threshold,
                      g=10.0,
                      sg=5,
                      sb=5,
                      b=1,
                      learning_rate=0.001,
                      phi=0.4028566874373059,
                      epsilon=0.003955266345682853,
                      retention_rate=1):
    env = ElFarolEnv(
        n_agents=n_agents,
        init_capacity=init_capacity,
        g=g,
        sg=sg,
        sb=sb,
        b=b)
    agents = []
    for i in range(0, n_agents):
        agents.append(
            ErevRothAgent(action_space=env.action_space,
                          config={
                              "init_mean": 1,
                              "init_std": 0,
                              "learning_rate": learning_rate,  # Reward multiplier
                              "phi": phi,
                              "epsilon": epsilon,
                              "retention_rate": retention_rate
                          }))
    return simulate(agents, env, visualise, simulation)


def simulate_egreedy(visualise=False,
                     simulation=random_threshold_changes,
                     g=20,
                     sg=9.38,
                     sb=2.739760371150226,
                     b=1.6782890889750117,
                     learning_rate=0.8769506278547066,
                     retention_rate=0.27681798313910244,
                     initial_epsilon=0.48904103108688907,
                     epsilon_decay=0.014380608025918491,
                     final_epsilon=0):
    env = ElFarolEnv(
        n_agents=n_agents,
        init_capacity=init_capacity,
        g=g,
        sg=sg,
        sb=sb,
        b=b)
    agents = []
    for i in range(0, n_agents):
        agents.append(
            EGreedyAgent(action_space=env.action_space,
                         config={
                             "learning_rate": learning_rate,  # Reward multiplier
                             "retention_rate": retention_rate,  # Forget some past experience
                             "initial_epsilon": initial_epsilon,  # Exploration probability
                             "epsilon_decay": epsilon_decay,  # Exploration reduction over time
                             "final_epsilon": final_epsilon  # Final exploration probability
                         }))
    return simulate(agents, env, visualise, simulation)


if __name__ == '__main__':
    print(simulate_egreedy(True, one_threshold_change))
