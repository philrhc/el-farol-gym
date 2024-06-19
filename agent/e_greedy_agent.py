import random
from collections import defaultdict

import numpy as np


class EGreedyAgent:
    def __init__(
        self,
        observation_space,
        action_space,
        learning_rate: float = 0.5,
        initial_epsilon: float = 0.2,
        epsilon_decay: float = 0.001,
        final_epsilon: float = 0.01,
        discount_factor: float = 0.95,

    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.q = defaultdict(lambda: np.zeros(action_space.n))
        self.action_space = action_space
        self.observation_space = observation_space
        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

    def act(self) -> int:
        a = None
        if np.random.random() < self.epsilon:
            a = self.action_space.sample()
        else:
            a = int(np.argmax(self.q))
        self.previous_action = a
        return a

    def learn(self, reward: float):
        self.q[self.previous_action] += self.lr * reward
        self.decay_epsilon()

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)