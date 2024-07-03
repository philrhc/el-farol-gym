from __future__ import print_function

from gymnasium import Env
from gymnasium.spaces import Discrete
import matplotlib.pyplot as plt
import numpy as np


class ElFarolEnv(Env):
    def __init__(self, n_agents=100, capacity=60, g=10, sg=5, sb=1, b=1):
        if g < sg or sg < sb or sb < b:
            raise Exception("rewards must be ordered g > sg > sb > b")

        self.n_agents = n_agents
        self.action_space = Discrete(2)
        self.reward_range = (b, g)
        self.capacity = capacity

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
        self.attendances = []
        self.capacities = []

    def modify_capacity(self, change):
        self.capacity = int(self.capacity + self.capacity * change)

    def step(self, action):
        n_attended = sum(action)
        reward = [self.reward_func(a, n_attended) for a in action]
        self.attendances.append(n_attended)
        self.capacities.append(self.capacity)
        return n_attended, reward, False, ()

    def plot_attendance_and_capacity(self):
        t = np.arange(0.0, 10_000, 1)
        fig, axs = plt.subplots(2, 1, layout='constrained')
        axs[0].plot(t, self.attendances)
        axs[0].plot(t, self.capacities)
        axs[0].set(xlabel='timesteps', ylabel="number of agents", title="Attendance/Capacity")
        axs[0].grid()

        mse = ((np.array(self.attendances) - np.array(self.capacities)) ** 2)
        axs[1].plot(t, mse)
        axs[1].set(title="Squared Error", xlabel='timesteps')
        axs[1].grid()

        plt.show()