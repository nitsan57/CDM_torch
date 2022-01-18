"""Demonstrates basic PDDLGym usage with random action sampling
"""
import matplotlib  # matplotlib.use('agg')

from Agents.RL_agents import rl_agent
from Agents.RL_agents.q_learning_agents import KERAS_DQN, Q_LEARNING
from constants import *

matplotlib.use('agg')  # For rendering

from pddlgym.pddlgym.utils import run_demo
import pddlgym.pddlgym as pddlgym
import imageio


def demo_random(env_name, render=True, problem_index=0, verbose=True):
    env = pddlgym.make("PDDLEnv{}-v0".format(env_name.capitalize()))
    env.fix_problem_index(problem_index)
    policy = lambda s: env.action_space.sample(s)
    video_path = "/tmp/{}_random_demo.mp4".format(env_name)
    run_demo(env, policy, render=render, verbose=verbose, seed=0,
             video_path=video_path)


def run_all(render=True, verbose=True):
    ## Some probabilistic environments
    demo_random("explodingblocks", render=render, verbose=verbose)
    # demo_random("tireworld", render=render, verbose=verbose)
    # demo_random("river", render=render, verbose=verbose)

    ## Some deterministic environments
    # demo_random("sokoban", render=render, verbose=verbose)
    # demo_random("gripper", render=render, verbose=verbose)
    # demo_random("rearrangement", render=render, problem_index=6, verbose=verbose)
    # demo_random("minecraft", render=render, verbose=verbose)
    # demo_random("blocks", render=render, verbose=verbose)
    # demo_random("blocks_operator_actions", render=render, verbose=verbose)
    # demo_random("quantifiedblocks", render=render, verbose=verbose)
    # demo_random("fridge", render=render, verbose=verbose)


# def create_env(env_name, operators_as_actions=False):
#     env = pddlgym.make("PDDLEnv{}-v0".format(env_name.capitalize()), operators_as_actions=operators_as_actions)
#     env.fix_problem_index(1)
#     obs = env.reset()
#     env_action_space = env.action_space
#     actions = set()
#     for i in range(100):
#         actions.add(env.sample_action_space(obs))
#     num_states = 600000
#     setattr(env_action_space, "n", len(env.action_space._all_ground_literals))
#     setattr(env, "actions", env.action_space._all_ground_literals)
#     setattr(env, "num_states", num_states)
#     return env


if __name__ == '__main__':
    env_name = "sokoban"
    # simple usage example
    # env = create_env(env_name, operators_as_actions=False)
    # obs = env.reset()
    # img = env.render()
    # imageio.imsave("frame0.png", img)  # Saved in \PDDLgym\pddlgym\pddlgym\
    # # action = env.action_space.sample(obs) # move(dir-down:direction), move(dir-left:direction), move(dir-right:direction), move(dir-up:direction)
    # actions = [3, 1, 1, 0, 0, 0]
    # for i, action in enumerate(actions):
    #     obs, reward, done, debug_info = env.step(action)
    #     img = env.render()
    #     imageio.imsave(f"frame{i + 1}.png", img)  # Saved in \PDDLgym\pddlgym\pddlgym\

    # # agent creation and training
    # env = create_env(env_name, operators_as_actions=False)
    # agent_name = Q_LEARNING
    # num_of_episodes = 500
    # agent = rl_agent.create_agent(env, agent_name)
    # train_result = rl_agent.run(agent, num_of_episodes, method=TRAIN)
    # rl_agent.run(agent, num_of_episodes, method=EVALUATE)
    # print(train_result)
    # print("DONE!")
