import random

import numpy as np

from agent.e_greedy import EGreedyAgent
from agent.simple_erev_roth import SimpleErevRothAgent
from agent.erev_roth import ErevRothAgent
from environment import ElFarolEnv, MultipleBarsEnv

iterations = 10_000
n_agents = 100
init_capacity = 70
capacity_change_chance = 0.2
capacity_change_limit = 0.3


def no_capacity_change(env, i):
    return


def one_change(env, i):
    if env.i == int(iterations / 2):
        env.modify_capacity(50, i)


def random_change(env, i):
    if env.i % 50 == 0 and env.i > 0:
        if random.random() < capacity_change_chance:
            change = (random.random() - 0.5) * capacity_change_limit
            env.modify_capacity_by_percentage(change, i)


def iterate(agents, env):
    actions = [a.act() for a in agents]
    transposed = np.transpose(actions)
    step = env.step(transposed)
    for agent_index, agent in enumerate(agents):
        reward_array = []
        for observations, rewards, _, _, in step:
            reward_array.append(rewards[agent_index])
        agent.learn(reward_array)
    return actions


def simulate_erevroth(visualise=False,
                      capacity_change_func=random_change,
                      g=10.0,
                      s=5,
                      b=1,
                      learning_rate=0.001,
                      phi=0.4028566874373059,
                      epsilon=0.003955266345682853,
                      retention_rate=1):
    env = ElFarolEnv(
        n_agents=n_agents,
        init_capacity=init_capacity,
        g=g,
        s=s,
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
    for i in range(0, iterations):
        iterate(agents, env)
    if visualise:
        env.plot_attendance_and_capacity(iterations)
    return env.mse()


def simulate_egreedy(visualise=False,
                     capacity_change_func=random_change,
                     g=10,
                     s=5,
                     b=1,
                     learning_rate=1,
                     retention_rate=0.02006827976496192,
                     initial_epsilon=0.5,
                     epsilon_decay=0.00698608772471109,
                     final_epsilon=0.17041582725849416):
    env = ElFarolEnv(
        n_agents=n_agents,
        init_capacity=init_capacity,
        g=g,
        s=s,
        b=b,
        capacity_change=capacity_change_func)
    agents = []
    for i in range(0, n_agents):
        agents.append(EGreedyAgent(action_space=env.action_space,
                                   config={
                                       "learning_rate": learning_rate,  # Reward multiplier
                                       "retention_rate": retention_rate,  # Forget some past experience
                                       "initial_epsilon": initial_epsilon,  # Exploration probability
                                       "epsilon_decay": epsilon_decay,  # Exploration reduction over time
                                       "final_epsilon": final_epsilon  # Final exploration probability
                                   }))
    for i in range(0, iterations):
        iterate(agents, env)
    if visualise:
        env.plot_attendance_and_capacity(iterations)
    return env.mse()


def multiple_bars(visualise=False,
                  g=10,
                  s=5,
                  b=1,
                  learning_rate=1,
                  retention_rate=0.02006827976496192,
                  initial_epsilon=0.5,
                  epsilon_decay=0.00698608772471109,
                  final_epsilon=0.17041582725849416):
    env = MultipleBarsEnv(
        n_agents=n_agents,
        init_capacity=[70, 15],
        g=g,
        s=s,
        b=b,
        capacity_change=[no_capacity_change, no_capacity_change])
    agents = []
    for i in range(0, n_agents):
        agents.append(EGreedyAgent(action_space=env.action_space,
                                   config={
                                       "learning_rate": learning_rate,  # Reward multiplier
                                       "retention_rate": retention_rate,  # Forget some past experience
                                       "initial_epsilon": initial_epsilon,  # Exploration probability
                                       "epsilon_decay": epsilon_decay,  # Exploration reduction over time
                                       "final_epsilon": final_epsilon  # Final exploration probability
                                   }))
    for i in range(0, iterations):
        iterate(agents, env)
    env.plot_attendance_and_capacity(iterations)
    return env.mse()


if __name__ == '__main__':
    print(multiple_bars())
