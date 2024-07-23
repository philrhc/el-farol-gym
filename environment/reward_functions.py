class ThresholdRewardFunc:
    def __init__(self, g, s, b):
        def fn(action, n_attended, capacity):
            if n_attended <= capacity:
                return g
            else:
                return b

        self.fn = fn


class AttendanceRewardFunc:
    def __init__(self, g, s, b):
        def fn(action, n_attended, capacity):
            if action == 0:
                return s
            elif n_attended <= capacity:
                return g
            else:
                return b

        self.fn = fn