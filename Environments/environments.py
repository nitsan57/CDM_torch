# from .taxi import SingleTaxiEnv
# from .frozen import FrozenLakeEnv


def register_env(env):
    name = env.__name__
    all_envs[name] = env

def get_all_avail_envs():
    all_names  = []
    for name in all_envs:
        all_names.append(name)
    return all_names

all_envs = {}