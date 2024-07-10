import gymnasium
import numpy as np

import random


class ErevRothAgent(object):
    def __init__(self, action_space: gymnasium.Space, config):
        self.prev_actions = []
        self.action_space = action_space
        self.config = config
        self.q = np.ones((action_space.shape[0], 2))

    def act(self):
        actions = []
        for i in range(self.q.shape[0]):
            total = sum([self.q[i][a] for a in range(2)])
            r = random.random()
            cum = 0
            for i, p in enumerate(self.q[i]):
                cum += p / total
                if r < cum:
                    actions.append(i)
                    break
        self.prev_actions = actions
        return actions

    def learn(self, reward):
        for i, each_reward in enumerate(reward):
            other_action = 1 - self.prev_actions[i]
            self.q[i][self.prev_actions[i]] = ((1 - self.config["phi"]) * self.q[i][self.prev_actions[i]]
                                               + each_reward * (1 - self.config["epsilon"]))
            self.q[i][other_action] = (1 - self.config["phi"]) * self.q[i][other_action] + each_reward * self.config["epsilon"]
