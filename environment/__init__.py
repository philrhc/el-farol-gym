from gym.envs.registration import register
from environment.el_farol_env import ElFarolEnv
from environment.pure_nash_equilibria import FuzzyPureNash
from environment.multiple_bars_env import MultipleBarsEnv

register(
    id='ElFarolEnv-v0',
    entry_point='gym.envs.multi_agent:ElFarolEnv'
    #,
    #timestep_limit=200,
    #local_only=True
)

