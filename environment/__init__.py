from gym.envs.registration import register
from environment.pure_nash_equilibria import FuzzyPureNash
from environment.el_farol import MultipleBarsEnv
from environment.reward_functions import AttendanceRewardFunc, ThresholdRewardFunc

register(
    id='ElFarolEnv-v0',
    entry_point='gym.envs.multi_agent:ElFarolEnv'
    #,
    #timestep_limit=200,
    #local_only=True
)

