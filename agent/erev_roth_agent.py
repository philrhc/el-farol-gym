import random
from collections import defaultdict

from gymnasium.spaces import Discrete


class ErevRothAgent(object):
    def __init__(self, observation_space, action_space, **userconfig):
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = {
            "init_mean": 1.0,  # Initialize Q values with this mean
            "init_std": 0.0,  # Initialize Q values with this standard deviation
            "learning_rate": 0.5,
        }
        self.config.update(userconfig)
        self.q = defaultdict(lambda: random.normalvariate(self.config["init_mean"], self.config["init_std"]))

    def act(self):
        # replace with numpy
        total = sum([self.q[a] for a in range(0, self.action_space.n)])
        r = random.random()
        cum = 0
        for a, p in self.q.items():
            cum += p / total
            if r < cum:
                self.prev_action = a
                return a
        raise Exception("No value selected")

    def learn(self, reward):
        self.q[self.prev_action] += reward * self.config["learning_rate"]
        for key in self.q:
            self.q[key] *= .99
