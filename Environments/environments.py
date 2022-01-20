from collections import defaultdict


def register_env(env):
    name = env.__name__
    train_envs[name] = env


def get_all_avail_envs():
    all_names  = []
    for name in train_envs:
        all_names.append(name)
    return all_names


def register_test_env(env, domain_name, difficulty):
    name = env.__name__
    test_envs[domain_name][difficulty][name] = env


def get_all_avail_test_envs(domain_name, difficulty):
    all_names  = []
    for name in test_envs[domain_name][difficulty]:
        all_names.append(name)
    return all_names


train_envs = {}
test_envs = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: dict)))

#defaultdict(defaultdict(defaultdict(dict)))
