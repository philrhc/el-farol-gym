import gymnasium
import numpy as np
import numpy.random


class EGreedyAgent(object):
    def __init__(self, action_space: gymnasium.Space, config):
        self.action_space = action_space
        self.config = config
        self.q = np.zeros((action_space.shape[0], 2))
        self.epsilon = self.config["initial_epsilon"]

    def act(self):
        a = None
        if numpy.random.random() < self.epsilon or np.count_nonzero(self.q) == 0:
            a = numpy.ndarray.flatten(self.action_space.sample())
        else:
            a = [np.argmax(self.q[i]) for i in range(self.q.shape[0])]
        self.prev_action = a
        return a

    def learn(self, reward):
        for index, each in enumerate(reward):
            previous_q = self.q[index][self.prev_action[index]]
            self.q[index][self.prev_action[index]] = previous_q + self.config["retention_rate"] * (each - previous_q)
            self.decay_epsilon()

    def decay_epsilon(self):
        self.epsilon = max(self.config["final_epsilon"], self.epsilon - self.config["epsilon_decay"])
