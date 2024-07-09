import random
from collections import defaultdict


class ErevRothAgent(object):
    def __init__(self, action_space, config):
        self.action_space = action_space
        self.config = config
        self.q = defaultdict(lambda: random.normalvariate(self.config["init_mean"], self.config["init_std"]))

    def act(self):
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
        other_action = 1 - self.prev_action
        self.q[self.prev_action] = (1 - self.config["phi"]) * self.q[self.prev_action] + reward * (1 - self.config["epsilon"])
        self.q[other_action] = (1 - self.config["phi"]) * self.q[other_action] + reward * self.config["epsilon"]
