from __future__ import print_function

import gymnasium as gym
import numpy
from gymnasium.vector import VectorEnv
from gymnasium.spaces import Discrete, Box
import matplotlib.pyplot as plt
import numpy as np


def squared(a, b):
    return np.array(a) - np.array(b) ** 2


class MultipleBarsEnv(VectorEnv):
    def __init__(self,
                 n_agents,
                 init_capacity,
                 capacity_change,
                 g=10,
                 s=5,
                 b=1):
        observation_space = Box(low=0, high=n_agents, dtype=np.int8)
        action_space = Box(low=0, high=1, dtype=np.int8)
        super().__init__(len(init_capacity), observation_space, action_space)
        if len(init_capacity) != len(capacity_change):
            raise Exception("init capacities and change not equal")
        self.i = 0
        self.n_agents = n_agents
        self.capacity = init_capacity
        self.capacity_change = capacity_change
        self.attendances = [[] for _ in range(len(init_capacity))]
        self.capacities = [[] for _ in range(len(init_capacity))]

        def reward_func(action, n_attended, capacity):
            if action == 0:
                return s
            elif n_attended <= capacity:
                return g
            else:
                return b

        self.reward_func = reward_func

    def modify_capacity_by_percentage(self, index, percentage_change):
        self.capacity[index] = int(self.capacity + self.capacity * percentage_change)

    def modify_capacity(self, index, new):
        self.capacity[index] = new

    def step(self, actions):
        self.i += 1
        observations = []
        for i in range(self.num_envs):
            n_attended = sum(actions[i])
            reward = [self.reward_func(a, n_attended, self.capacity[i]) for a in actions[i]]
            self.attendances[i].append(n_attended)
            self.capacities[i].append(self.capacity[i])
            self.capacity_change[i](self, i)
            observations.append((n_attended, reward, False, ()))
        return observations

    def mse(self):
        total = 0
        for i in range(self.num_envs):
            squared_error = squared(self.attendances[i], self.capacities[i])
            sum = 0
            for each in squared_error:
                sum += each
            total += sum / len(squared_error)
        return total

    def plot_attendance_and_capacity(self, iterations):
        t = np.arange(0, iterations, 1)
        fig, axs = plt.subplots(nrows=self.num_envs, ncols=2, layout='constrained')
        for i in range(self.num_envs):
            axs[i][0].plot(t, self.attendances[i])
            axs[i][0].plot(t, self.capacities[i])
            axs[i][0].set(xlabel='timesteps', ylabel="number of agents", title="Attendance/Capacity")
            axs[i][0].grid()

            squared_error = squared(self.attendances[i], self.capacities[i])
            axs[i][1].plot(t, squared_error)
            axs[i][1].set(title="Squared Error", xlabel='timesteps')
            axs[i][1].grid()

        plt.show()
