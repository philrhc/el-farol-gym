import random


def no_capacity_change(env, i):
    return


class OneChange:
    def __init__(self, iterations):
        def func(env, i):
            if env.i == int(iterations / 2):
                env.modify_capacity(50, i)
        self.func = func


class RandomChanges:
    def __init__(self, change_chance, change_limit):
        def func(env, i):
            if env.i % 50 == 0 and env.i > 0:
                if random.random() < change_chance:
                    change = (random.random() - 0.5) * change_limit
                    env.modify_capacity_by_percentage(change, i)
        self.func = func
