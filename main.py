
from Environments.environments import all_envs, get_all_avail_envs
from Models import fc, rnn
from Agents.dqn_agent import DQN_Agent
from Agents.ppo_agent import PPO_Agent

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import utils
import gym
# from Curriculum_managers

# def get_env(env_name):
#     env = all_envs[env_name](random_reset_loc=True)
#     n_actions = env.action_space.n
#     obs_shape = env.observation_space['image'].shape
#     env.dummy_init()
#     return env, obs_shape, n_actions

def get_env(env_name):
    env = gym.envs.make("CartPole-v1")
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape
    return env, obs_shape, n_actions


def main():
    env_names = get_all_avail_envs()
    env_names
    device = utils.init_torch()
    env, obs_shape, n_actions = get_env('SingleTaxiEnv')
    agent = DQN_Agent(obs_shape, n_actions, device=device, batch_size=64, max_mem_size=10**5, exploration_epsilon=0.3, eps_dec=0, rnn=True, model=rnn.RNN)
    # agent = PPO_Agent(env, obs_shape, n_actions, device=device, batch_size=64, max_mem_size=10**5, lr=0.001)
    agent.train_episodial(env, 20000)

if __name__ == "__main__":
    main()