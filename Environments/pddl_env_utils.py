from pddlgym import pddlgym

def get_pddl_env(env_name, operators_as_actions=False, env_idx=0):
    env = pddlgym.make("PDDLEnv{}-v0".format(env_name.capitalize()), operators_as_actions=operators_as_actions)
    env.fix_problem_index(env_idx)
    obs = env.reset()
    obs_space=obs.shape
    env_action_space = env.action_space
    actions = set()
    for i in range(1):
        actions.add(env.sample_action_space(obs))
    num_states = 600000
    setattr(env_action_space, "n", len(env.action_space._all_ground_literals))
    setattr(env, "actions", env.action_space._all_ground_literals)
    setattr(env, "num_states", num_states)
    setattr(env.env, "n", len(env.action_space._all_ground_literals))
    setattr(env.env, "actions", env.action_space._all_ground_literals)
    setattr(env.env, "num_states", num_states)
    return env,obs_space, env_action_space.n