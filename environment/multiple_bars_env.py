from __future__ import print_function

import gymnasium as gym
import numpy
from gymnasium.vector import VectorEnv
from gymnasium.spaces import Discrete, Box
import matplotlib.pyplot as plt
import numpy as np


class MultipleBarsEnv(VectorEnv):
    def __init__(self, n_agents, init_capacity, g, s, b, capacity_change):
        observation_space = Box(low=0, high=n_agents, dtype=np.int8)
        action_space = Box(low=0, high=1, dtype=np.int8)
        super().__init__(len(init_capacity), observation_space, action_space)
        if len(init_capacity) != len(capacity_change):
            raise Exception("init capacities and change not equal")
        self.i = 0
        self.n_agents = n_agents
        self.capacity = init_capacity
        self.capacity_change = capacity_change
        self.attendances = []
        self.capacities = []

        def reward_func(action, n_attended):
            if action == 0:
                return s
            elif n_attended <= self.capacity:
                return g
            else:
                return b

        self.reward_func = reward_func

    def modify_capacity_by_percentage(self, percentage_change):
        self.capacity = int(self.capacity + self.capacity * percentage_change)

    def modify_capacity(self, new):
        self.capacity = new

    def step(self, actions):
        self.i += 1
        observations = []
        step_attendances = []
        step_capacities = []
        for i in range(self.num_envs):
            n_attended = sum(actions[i])
            reward = [self.reward_func(a, n_attended) for a in actions[i:]]
            self.attendances.append(n_attended)
            self.capacities.append(self.capacity)
            self.capacity_change[i](self)
            observations.append((n_attended, reward, False, ()))
        self.attendances.append(step_attendances)
        self.capacities.append(step_capacities)
        return observations

    def mse(self):
        squared_error = ((np.array(self.attendances) - np.array(self.capacities)) ** 2)
        sum = 0
        for each in squared_error:
            sum += each
        return sum / len(squared_error)

    def plot_attendance_and_capacity(self, iterations):
        t = np.arange(0.0, iterations, 1)
        fig, axs = plt.subplots(2, 1, layout='constrained')
        axs[0].plot(t, self.attendances)
        axs[0].plot(t, self.capacities)
        axs[0].set(xlabel='timesteps', ylabel="number of agents", title="Attendance/Capacity")
        axs[0].grid()

        squared_error = ((np.array(self.attendances) - np.array(self.capacities)) ** 2)
        axs[1].plot(t, squared_error)
        axs[1].set(title="Squared Error", xlabel='timesteps')
        axs[1].grid()

        plt.show()