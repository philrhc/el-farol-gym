from __future__ import print_function

from gymnasium import Env
from gymnasium.spaces import Discrete
import matplotlib.pyplot as plt
import numpy as np


class ElFarolEnv(Env):
    def __init__(self, n_agents, init_capacity, g, sg, sb, b):
        self.n_agents = n_agents
        self.action_space = Discrete(2) # used by agents when sampling action space
        self.capacity = init_capacity
        self.attendances = []
        self.capacities = []

        def reward_func(action, n_attended):
            if action == 0:
                if n_attended >= self.capacity:
                    return sg
                if n_attended < self.capacity:
                    return sb
            elif n_attended <= self.capacity:
                return g
            else:
                return b

        self.reward_func = reward_func

    def modify_capacity_by_percentage(self, percentage_change):
        self.capacity = int(self.capacity + self.capacity * percentage_change)

    def modify_capacity(self, new):
        self.capacity = new

    def step(self, action):
        n_attended = sum(action)
        reward = [self.reward_func(a, n_attended) for a in action]
        self.attendances.append(n_attended)
        self.capacities.append(self.capacity)
        return n_attended, reward, False, ()

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