import math


class ThresholdRewardFunc:
    def __init__(self, g, s, b):
        def fn(action, n_attended, capacity):
            if n_attended <= capacity:
                return 0
            else:
                return 1

        self.fn = fn


class ExponentialDecayRewardFunc:
    def __init__(self, g, s, b):
        def fn(action, n_attended, capacity):
            if n_attended <= capacity:
                return 0
            return math.exp(-abs(n_attended - capacity))

        self.fn = fn


class ElFarolRewardFunc:
    def __init__(self, g, s, b):
        def fn(action, n_attended, capacity):
            if action == 0:
                return s
            elif n_attended <= capacity:
                return g
            else:
                return b

        self.fn = fn
