import gymnasium
import numpy as np
import numpy.random


class EGreedyAgent(object):
    def __init__(self, action_space: gymnasium.Space, config):
        self.action_space = action_space
        self.config = config
        self.q = np.zeros(2)
        self.epsilon = self.config["initial_epsilon"]

    def act(self):
        a = None
        if numpy.random.random() < self.epsilon or np.count_nonzero(self.q) == 0:
            a = self.action_space.sample()
        else:
            a = np.argmax(self.q)
        self.prev_action = a
        return a

    def learn(self, reward):
        previous_q = self.q[self.prev_action]
        self.q[self.prev_action] = previous_q + self.config["retention_rate"] * (reward - previous_q)
        self.decay_epsilon()

    def decay_epsilon(self):
        self.epsilon = max(self.config["final_epsilon"], self.epsilon - self.config["epsilon_decay"])
