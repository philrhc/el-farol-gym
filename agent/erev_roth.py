import gymnasium
import numpy as np
import random


class ErevRothAgent(object):
    def __init__(self, action_space: gymnasium.Space, config):
        self.action_space = action_space
        self.config = config
        self.q = np.ones(2)

    def act(self):
        total = sum([self.q[a] for a in range(2)])
        r = random.random()
        cum = 0
        for i, p in enumerate(self.q):
            cum += p / total
            if r < cum:
                action = i
                break
        self.prev_action = action
        return action

    def learn(self, reward):
        other_action = 1 - self.prev_action
        self.q[self.prev_action] = ((1 - self.config["phi"]) * self.q[self.prev_action]
                                           + reward * (1 - self.config["epsilon"]))
        self.q[other_action] = (1 - self.config["phi"]) * self.q[other_action] + reward * self.config["epsilon"]
