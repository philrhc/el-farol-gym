import random


def no_capacity_change(env, i):
    return


class OneChange:
    def __init__(self, iterations, final_value):
        def func(env, i):
            if env.i == int(iterations / 2):
                env.modify_capacity(i, final_value)
        self.func = func


class RandomChanges:
    def __init__(self, chance, limit):
        def func(env, i):
            if env.i % 50 == 0 and env.i > 0:
                if random.random() < chance:
                    change = (random.random() - 0.5) * limit
                    env.modify_capacity_by_percentage(i, change)
        self.func = func
