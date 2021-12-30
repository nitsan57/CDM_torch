from pickle import DUP

from .curriculum_manager import Curriculum_Manager
from copy import deepcopy
import torch
import os
import numpy as np
from Agents.agent_utils import ParallelEnv




class PAIRED_Curriculum(Curriculum_Manager):
    def __init__(self, abstract_env, trainee, teacher_agent, n_episodes = 100, n_iters=50000) -> None:
        super().__init__(abstract_env, trainee)
        self.antagonist = deepcopy(trainee)
        self.random_z_space = 10
        
        if type(self.teacher_obs_space) != dict:
            self.teacher_obs_space = dict()
            self.teacher_obs_space['data'] = abstract_env.get_observation_space()
        self.teacher_obs_space['random_z'] = self.random_z_space
        self.teacher = teacher_agent

        # self.model_agent = teacher_agent(input_shape=self.teacher_obs_space, out_shape=self.teacher_action_space).to(self.device)
        # self.actor_model = lambda x : Categorical(logits=F.softmax(self.policy_nn(x), dim=1))
        
        
        super().__init__(trainee, n_episodes, n_iters)

    def get_next_env_block(self, obs):
        random_z  = torch.rand(self.random_z_space, device=self.device)
        obs['random_z'] = random_z
        # or act
        teacher_action = self.teacher.act(obs)
        return teacher_action



    def create_env(self, number_of_envs=1, teacher_eval_mode=False):
    # obs = self.abstract_env.clear_env()
        if teacher_eval_mode:
            self.teacher.set_eval_mode()
        a_env = self.abstract_env
        if number_of_envs > 1:
            a_env = ParallelEnv(a_env)
        self.teacher.collect_episode_obs(a_env, max_episode_len=self.teacher_max_steps, env_funcs={"step":"step_generator", "reset":"clear_env"})
        return a_env

        
    def save_models(self):
        save_dir = "./paired_agent_weights"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.trainee.save_agent(save_dir)
        self.antagonist.save_agent(save_dir)
        self.teacher.save_agent(save_dir)
        

    def set_agent_to_train_mode(self):
        self.teacher.set_train_mode()
        self.trainee.set_train_mode()
        self.antagonist.set_train_mode()

    def teach(self, n_iters, n_steps_p_env=10000):
        self.set_agent_to_train_mode()

        number_episodes_to_collect_in_parallel = 4 # same as paired paper
        self.trainee.set_num_parallel_env(number_episodes_to_collect_in_parallel)
        self.antagonist.set_num_parallel_env(number_episodes_to_collect_in_parallel)

        for iter in range(n_iters):
            env = self.create_env(train_teacher=True)

            env = ParallelEnv(env, number_episodes_to_collect_in_parallel)
            
            max_episode_steps = env.get_max_episode_steps()
            
            n_steps_collected = 0
            while n_steps_collected < n_steps_p_env:
                trainee_rewards = self.trainee.collect_episode_obs(env, max_episode_len=max_episode_steps) #collect a single episode experience - controled in num_env_parallel in each agent
                antagonist_rewards = self.antagonist.collect_episode_obs(env, max_episode_len=max_episode_steps)
                episodes_collected = len(trainee_rewards)
                curr_steps_collected = np.sum([len(r) for r in trainee_rewards])

                trainee_avg_r = np.mean(trainee_rewards)
                trainee_max_r = np.max(trainee_rewards)
                anta_avg_r = np.mean(antagonist_rewards)
                anta_max_r = np.max(antagonist_rewards)

                #Change agents update... as paired paper states..
                # # update rewards:
                reward_buffer_index = self.trainee.experience.reward_index

                #sample last experiences
                t_exp = self.trainee.experience.get_last_episodes(number_episodes_to_collect_in_parallel) # maybe change to 1 on ppo?
                a_exp = self.antagonist.get_last_episodes(number_episodes_to_collect_in_parallel) # maybe change to 1 on ppo?

                t_exp[reward_buffer_index] = t_exp[reward_buffer_index] - (anta_max_r) / max_episode_steps
                a_exp[reward_buffer_index] = a_exp[reward_buffer_index] - (trainee_max_r) / max_episode_steps

                self.trainee.update_policy(t_exp)
                self.antagonist.update_policy(a_exp)

                n_steps_collected += curr_steps_collected

            # train teacher_model

            teacher_reward = anta_max_r - trainee_avg_r

            teacher_exp = self.teacher.experience.get_last_episodes(number_episodes_to_collect_in_parallel) 
            teacher_exp[reward_buffer_index] = teacher_reward
            self.teacher.update_policy(teacher_exp)
            self.teacher.experience.clear()
            self.trainee.experience.clear()
            self.antagonist.experience.clear()


    # return self.abstract_env #maybe copy to avoid dynamic changes

    # def create_env(self, train_teacher=False):
    #     # obs = self.abstract_env.clear_env()
    #     self.teacher.collect_episode_obs(self.abstract_env, max_episode_len=None, env_funcs={"step":"step_generator", "reset":"clear_env"})
    #     # created env is the last observation:
    #     self.teacher.experiences
    #     created_env = 

    #     # # self.teacher.experience.append(observations, actions, rewards_x, dones, next_observations)
    #     # all_obs = []
    #     # # all_rewards = []
    #     # # all_dones = []

    #     # for i in range(self.abstract_env.generator_max_steps):
    #     #     teacher_action = self.get_next_env_block(obs)
    #     #     obs, reward, done, _ = self.abstract_env.step_generator(teacher_action) 
    #     #     # all_rewards.append(reward)
    #     #     # all_dones.append(done)
    #     #     if done:
    #     #         break
    #     #     if train_teacher:
    #     #         self.experience.append(observations, actions, rewards_x, dones, next_observations)


    #     # return self.abstract_env #maybe copy to avoid dynamic changes