import random
from collections import defaultdict

import numpy as np


class EGreedyAgent(object):
    def __init__(self, observation_space, action_space, **userconfig):
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = {
            "learning_rate": 0.5,
            "initial_epsilon": 0.5,
            "epsilon_decay": 0.01,
            "final_epsilon": 0.1
        }
        self.config.update(userconfig)
        self.q = [0, 0]
        self.epsilon = self.config["initial_epsilon"]

    def act(self):
        a = None
        if random.random() < self.epsilon or np.count_nonzero(self.q) == 0:
            a = self.action_space.sample()
        else:
            a = np.argmax(self.q)
        self.prev_action = a
        return a

    def learn(self, reward):
        self.q[self.prev_action] += reward * self.config["learning_rate"]

    def decay_epsilon(self):
        self.epsilon = max(self.config["final_epsilon"], self.epsilon - self.config["epsilon_decay"])