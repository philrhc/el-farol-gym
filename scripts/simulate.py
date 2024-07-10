import random

import numpy as np

from agent.e_greedy import EGreedyAgent
from agent.simple_erev_roth import SimpleErevRothAgent
from agent.erev_roth import ErevRothAgent
from environment import ElFarolEnv, MultipleBarsEnv
import capacity_changes
import hyperparams

iterations = 10_000
n_agents = 100
init_capacity = 70
capacity_change_chance = 0.2
capacity_change_limit = 0.3


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


def simulate(visualise=False,
             agent_type=EGreedyAgent,
             config=hyperparams.e_greedy_optimal):
    env = MultipleBarsEnv(
        n_agents=n_agents,
        init_capacity=[70, 15],
        capacity_change=[capacity_changes.no_capacity_change, capacity_changes.no_capacity_change])
    agents = [agent_type(action_space=env.action_space, config=config) for _ in range(0, n_agents)]
    [iterate(agents, env) for _ in range(0, iterations)]
    if visualise:
        env.plot_attendance_and_capacity(iterations)
    return env.mse()


if __name__ == '__main__':
    print(simulate(True, ErevRothAgent, hyperparams.erev_roth_optimal))
