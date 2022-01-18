
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
# from Curriculum_managers.paired_curriculum_no_regret_teacher_entropy import  PAIRED_Curriculum_no_regret_entropy
from Curriculum_managers.paired_curriculum_no_regulator import  Curriculum_Entropy_Only
import plotly.express as px

from pddlgym import pddlgym

def get_env_norm(env_name):
    env = all_envs[env_name](random_reset_loc=False)
    n_actions = env.action_space.n
    obs_shape = env.observation_space
    return env, obs_shape, n_actions

def get_env(env_name, operators_as_actions=False, env_idx=0):
    env = pddlgym.make("PDDLEnv{}-v0".format(env_name.capitalize()), operators_as_actions=operators_as_actions)
    env.fix_problem_index(env_idx)
    obs = env.reset()
    obs_space=(1,)
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

def main():
    env_names = get_all_avail_envs()
    env_names
    device = utils.init_torch()
    # gen_obs_shape = env.get_generator_observation_space().shape
    # gen_action_dim = env.get_generator_action_space().n
    import pdb
    pdb.set_trace()
    env, obs_shape, n_actions = get_env('sokoban')
    

    # env, obs_shape, n_actions = get_env_norm('MiniAdversarialEnv')

    # r_agent = DQN_Agent(obs_shape, n_actions, device=device, batch_size=64, max_mem_size=10**5,num_parallel_envs=4, lr=0.0001, model=rnn.RNN)
    # r_agent = DQN_Agent(obs_shape, n_actions, device=device, batch_size=64, max_mem_size=10**5,exploration_epsilon=0.5, lr=0.001, model=fc.FC)
    # r_agent.set_store_entropy(True)
    # r_agent.train_episodial(env, 2000)
    # print("=000000000000000000000000000000")
    # r_agent.train_episodial(env, 12000)


    # r_agent2 = DQN_Agent(obs_shape, n_actions, device=device, batch_size=64, max_mem_size=10**5, exploration_epsilon=0.3, eps_dec=0, lr=0.001, model=rnn.RNN)
    # t = r_agent2.train_episodial(env, 20000)

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

    # pe_agent = DQN_Agent(obs_shape, n_actions, device=device, batch_size=64, max_mem_size=10**5, exploration_epsilon=0.8, eps_dec=0, lr=0.001, model=rnn.RNN)
    # pe_teacher_agent = DQN_Agent(gen_obs_shape, gen_action_dim, device=device, batch_size=64, max_mem_size=10**5, exploration_epsilon=1, eps_dec=0.0001, lr=0.001, model=rnn.RNN)
    # pe_teacher = PAIRED_Curriculum_no_regret_entropy(env, teacher_agent=pe_teacher_agent ,trainee=pe_agent)
    # pe_rewards = pe_teacher.teach(n_iters=10000, n_episodes=8)


    # pen_agent = PPO_Agent(obs_shape, n_actions, device=device, batch_size=64, max_mem_size=10**5, lr=0.0001, model=rnn.RNN)
    # pen_teacher_agent = PPO_Agent(gen_obs_shape, gen_action_dim, device=device, batch_size=64, max_mem_size=10**5, lr=0.001, model=rnn.RNN)
    # pen_teacher = Curriculum_Entropy_Only(env, teacher_agent=pen_teacher_agent ,trainee=pen_agent)
    # pen_rewards = pen_teacher.teach(n_iters=20000, n_episodes=1)



    
    agent = PPO_Agent(obs_shape, n_actions, device=device, batch_size=64, max_mem_size=10**5,num_parallel_envs=64, lr=0.0001 ,model=rnn.RNN)
    train_rewards = agent.train_episodial(env, n_episodes=10000)
    # pee_agent = PPO_Agent(obs_shape, n_actions, device=device, batch_size=64, max_mem_size=10**5, lr=0.0001, model=rnn.RNN)
    # pee_teacher_agent = PPO_Agent(gen_obs_shape, n_actions, device=device, batch_size=64, max_mem_size=10**5, lr=0.0001, model=rnn.RNN)
    # pee_teacher = Curriculum_Entropy_Only(env, teacher_agent=pee_teacher_agent ,trainee=pee_agent)
    # pee_rewards = pee_teacher.teach(n_iters=30000, n_episodes=1)
    

if __name__ == "__main__":
    main()