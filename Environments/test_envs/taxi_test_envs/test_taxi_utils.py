from ast import arg
import random
from Environments.taxi import SingleTaxiEnv

def init_for_test(init_vector, random_env, **kwargs):
    def init():
        env = SingleTaxiEnv(**kwargs)
        if random_env:
            env.reset_random()
        else:
            env.clear_env()
            env.init_from_vec(init_vector)
        return env
    return init
