from __future__ import print_function
from gymnasium.vector import VectorEnv
from gymnasium.spaces import Box
from queue import SimpleQueue, Queue
import matplotlib.pyplot as plt
import numpy as np


def squared(a, b):
    return np.square(np.array(a) - np.array(b))


def initialise_reward_queue(n_agents, n_envs, delay):
    reward_queues = [SimpleQueue() for _ in range(n_envs)]
    for i in range(n_envs):
        for _ in range(delay):
            reward_queues[i].put(np.zeros(n_agents))
    return reward_queues


class MultipleBarsEnv(VectorEnv):
    def __init__(self,
                 n_agents,
                 init_capacity,
                 capacity_change,
                 reward_func,
                 reward_delay=0,
                 g=10,
                 s=5,
                 b=1):
        observation_space = Box(low=0, high=n_agents, dtype=np.int8)
        action_space = Box(low=0, high=1, dtype=np.int8)
        super().__init__(len(init_capacity), observation_space, action_space)
        if len(init_capacity) != len(capacity_change):
            raise Exception("init capacities and capacity change functions not equal size")
        n_envs = len(init_capacity)
        self.i = 0
        self.capacity = init_capacity
        self.capacity_change = capacity_change
        self.attendances = [[] for _ in range(n_envs)]
        self.capacities = [[] for _ in range(n_envs)]
        self.reward_func = reward_func(g, s, b).fn
        self.reward_queue = initialise_reward_queue(n_agents, n_envs, reward_delay)

    def modify_capacity_by_percentage(self, index, percentage_change):
        self.capacity[index] = int(self.capacity[index] + self.capacity[index] * percentage_change)

    def modify_capacity(self, index, new):
        self.capacity[index] = new

    def step(self, actions):
        self.i += 1
        observations = []
        for i in range(self.num_envs):
            n_attended = sum(actions[i])
            reward = [self.reward_func(a, n_attended, self.capacity[i]) for a in actions[i]]
            self.reward_queue[i].put(reward)
            observed_reward = self.reward_queue[i].get()
            observations.append((n_attended, observed_reward, False, ()))
            self.attendances[i].append(n_attended)
            self.capacities[i].append(self.capacity[i])
            self.capacity_change[i](self, i)
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

        def plot(attendance_capacity_graph, mse_graph, index):
            attendance_capacity_graph.plot(t, self.attendances[index])
            attendance_capacity_graph.plot(t, self.capacities[index])
            attendance_capacity_graph.set(xlabel='timesteps', ylabel="number of agents", title="Attendance/Capacity")
            attendance_capacity_graph.grid()

            squared_error = squared(self.attendances[index], self.capacities[index])
            mse_graph.plot(t, squared_error)
            mse_graph.set(title="Squared Error", xlabel='timesteps')
            mse_graph.grid()

        if self.num_envs == 1:
            plot(axs[0], axs[1], 0)
        else:
            for i in range(self.num_envs):
                plot(axs[i][0], axs[i][1], i)

        plt.show()
