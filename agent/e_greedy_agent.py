import random
import numpy as np
import numpy.random


class EGreedyAgent(object):
    def __init__(self, action_space, **userconfig):
        self.action_space = action_space
        self.config = {
            "learning_rate": 0.5,  # Reward multiplier
            "retention_rate": 0.8,    # Forget some past experience
            "initial_epsilon": 0.5,  # Exploration probability
            "epsilon_decay": 0.01,  # Exploration reduction over time
            "final_epsilon": 0.01  # Final exploration probability
        }
        self.config.update(userconfig)
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
        self.q[self.prev_action] += reward * self.config["learning_rate"]
        self.forget()
        self.decay_epsilon()

    def forget(self):
        for x in range(2):
            self.q[x] *= self.config["retention_rate"]

    def decay_epsilon(self):
        self.epsilon = max(self.config["final_epsilon"], self.epsilon - self.config["epsilon_decay"])
