from ast import arg
import random
from Environments.frozen import FrozenLakeEnv

def init_for_test(init_vector, **kwargs):
    def init():
        env = FrozenLakeEnv(**kwargs)
        env.clear_env()
        env.init_from_vec(init_vector)
        return env
    return init
