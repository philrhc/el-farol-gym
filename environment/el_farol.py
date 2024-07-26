from __future__ import print_function

import gym
from gymnasium.vector import VectorEnv
from gymnasium.spaces import Box, Discrete
from queue import SimpleQueue
import matplotlib.pyplot as plt
import numpy as np


def squared(a, b):
    return np.square(np.array(a) - np.array(b))


def initialise_reward_queue(n_agents, delay):
    reward_queue = SimpleQueue()
    for _ in range(delay):
        reward_queue.put(np.zeros(n_agents))
    return reward_queue


class MultipleBarsEnv(gym.Env):
    def __init__(self,
                 n_agents,
                 init_capacity,
                 capacity_change,
                 reward_func,
                 reward_delay=0,
                 g=10,
                 s=5,
                 b=1):
        self.observation_space = Box(low=0, high=n_agents, dtype=np.int8)
        self.action_space = Discrete(2)
        self.i = 0
        self.capacity = init_capacity
        self.capacity_change = capacity_change
        self.attendances = []
        self.capacities = []
        self.reward_func = reward_func(g, s, b).fn
        self.reward_queue = initialise_reward_queue(n_agents, reward_delay)

    def modify_capacity_by_percentage(self, percentage_change):
        self.capacity = int(self.capacity + self.capacity * percentage_change)

    def modify_capacity(self, new):
        self.capacity = new

    def step(self, actions):
        self.i += 1
        n_attended = sum(actions)

        # queue rewards to simulate information delay
        reward = [self.reward_func(a, n_attended, self.capacity) for a in actions]
        self.reward_queue.put(reward)
        observed_reward = self.reward_queue.get()

        # log observations for graphing
        self.attendances.append(n_attended)
        self.capacities.append(self.capacity)

        # change capacity to simulate variable inputs
        self.capacity_change(self, self.i)

        return (n_attended, observed_reward, False, ())

    def mse(self):
        total = 0
        squared_error = squared(self.attendances, self.capacities)
        sum = 0
        for each in squared_error:
            sum += each
        total += sum / len(squared_error)
        return total

    def plot_attendance_and_capacity(self, iterations):
        t = np.arange(0, iterations, 1)
        fig, axs = plt.subplots(nrows=1, ncols=2, layout='constrained')

        def plot(attendance_capacity_graph, mse_graph):
            attendance_capacity_graph.plot(t, self.attendances)
            attendance_capacity_graph.plot(t, self.capacities)
            attendance_capacity_graph.set(xlabel='timesteps', ylabel="number of agents", title="Attendance/Capacity")
            attendance_capacity_graph.grid()

            squared_error = squared(self.attendances, self.capacities)
            mse_graph.plot(t, squared_error)
            mse_graph.set(title="Squared Error", xlabel='timesteps')
            mse_graph.grid()

        plot(axs[0], axs[1])

        plt.show()
