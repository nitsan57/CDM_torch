
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
from Curriculum_managers.random_curriculum import Random_Curriculum
from Curriculum_managers.paired_curriculum import PAIRED_Curriculum
# from Curriculum_managers.paired_curriculum_extented import  PAIRED_Curriculum_entropy
from Curriculum_managers.paired_curriculum_no_regret import  PAIRED_Curriculum_no_regret
from Curriculum_managers.paired_curriculum_no_regret_teacher_entropy import  PAIRED_Curriculum_no_regret_entropy
import plotly.express as px

def get_env(env_name):
    env = all_envs[env_name](random_reset_loc=True)
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape
    return env, obs_shape, n_actions

# def get_env(env_name):
#     env = gym.envs.make("CartPole-v1")
#     n_actions = env.action_space.n
#     obs_shape = env.observation_space.shape
#     return env, obs_shape, n_actions


def main():
    env_names = get_all_avail_envs()
    env_names
    device = utils.init_torch()
    env, obs_shape, n_actions = get_env('MiniAdversarialEnv')
    gen_obs_shape = env.get_generator_observation_space().shape
    gen_action_dim = env.get_generator_action_space().n

    # agent = DQN_Agent(obs_shape, n_actions, device=device, batch_size=64, max_mem_size=10**5, exploration_epsilon=0.3, eps_dec=0, num_parallel_envs=64,  model=rnn.RNN)
    # agent = DQN_Agent(obs_shape, n_actions, device=device, batch_size=64, max_mem_size=10**5, exploration_epsilon=0.3, eps_dec=0, num_parallel_envs=64)

    # r_agent = DQN_Agent(obs_shape, n_actions, device=device, batch_size=64, max_mem_size=10**5, exploration_epsilon=0.3, eps_dec=0, lr=0.001, model=rnn.RNN)
    # train_rewards = r_agent.train_episodial(env, 40000)


    # r_teacher = Random_Curriculum(env, trainee=agent)
    # r_teacher.teach(1000)


    # agent = DQN_Agent(obs_shape, n_actions, device=device, batch_size=64, max_mem_size=10**5, exploration_epsilon=0.3, eps_dec=0, lr=0.001, model=rnn.RNN)
    # teacher_agent = DQN_Agent(gen_obs_shape, gen_action_dim, device=device, batch_size=64, max_mem_size=10**5, exploration_epsilon=0.3, eps_dec=0, lr=0.001, model=rnn.RNN)
    # p_teacher = PAIRED_Curriculum(env, teacher_agent=teacher_agent, trainee=agent)
    # # e = p_teacher.create_env()
    # p_teacher.teach(1000, 8)


    # p_agent = PPO_Agent(obs_shape, n_actions, device=device, batch_size=512, max_mem_size=10**5, lr=0.0001, model=rnn.RNN)
    # teacher_agent = PPO_Agent(gen_obs_shape, gen_action_dim, device=device, batch_size=512, max_mem_size=10**5, lr=0.0001, model=rnn.RNN)
    # p_teacher = PAIRED_Curriculum_entropy(env, teacher_agent=teacher_agent ,trainee=p_agent)
    # p_rewards = p_teacher.teach(n_iters=5000, n_episodes=8)

    # pnr_agent = DQN_Agent(obs_shape, n_actions, device=device, batch_size=64, max_mem_size=10**5, exploration_epsilon=0.8, eps_dec=0, lr=0.001, model=rnn.RNN)
    # teacher_agent_nr = DQN_Agent(gen_obs_shape, gen_action_dim, device=device, batch_size=64, max_mem_size=10**5, exploration_epsilon=1, eps_dec=0.0001, lr=0.001, model=rnn.RNN)
    # pnr_teacher = PAIRED_Curriculum_no_regret(env, teacher_agent=teacher_agent_nr ,trainee=pnr_agent)
    # pnr_rewards = pnr_teacher.teach(n_iters=5000, n_episodes=8)

    pe_agent = DQN_Agent(obs_shape, n_actions, device=device, batch_size=64, max_mem_size=10**5, exploration_epsilon=0.8, eps_dec=0, lr=0.001, model=rnn.RNN)
    pe_teacher_agent = DQN_Agent(gen_obs_shape, gen_action_dim, device=device, batch_size=64, max_mem_size=10**5, exploration_epsilon=1, eps_dec=0.0001, lr=0.001, model=rnn.RNN)
    pe_teacher = PAIRED_Curriculum_no_regret_entropy(env, teacher_agent=pe_teacher_agent ,trainee=pe_agent)
    pe_rewards = pe_teacher.teach(n_iters=20000, n_episodes=8)

if __name__ == "__main__":
    main()