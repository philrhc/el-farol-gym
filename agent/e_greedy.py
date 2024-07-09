import numpy as np
import numpy.random


class EGreedyAgent(object):
    def __init__(self, action_space, config):
        self.action_space = action_space
        self.config = config
        self.q = [0, 0]
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
        self.q[self.prev_action] = ((reward * self.config["learning_rate"] * self.config["retention_rate"])
                                    + (1 - self.config["retention_rate"]) * self.q[self.prev_action])
        self.decay_epsilon()

    def decay_epsilon(self):
        self.epsilon = max(self.config["final_epsilon"], self.epsilon - self.config["epsilon_decay"])
