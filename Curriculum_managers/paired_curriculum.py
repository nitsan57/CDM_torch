from .curriculum_manager import Curriculum_Manager
from copy import deepcopy
import torch
import os
import numpy as np
from Agents.agent_utils import ParallelEnv
from tqdm import tqdm
import functools
import operator


class PAIRED_Curriculum(Curriculum_Manager):
    def __init__(self, abstract_env, trainee, teacher_agent, n_episodes = 100, n_iters=50000) -> None:
        save_dir = "./paired_agent_weights/" + abstract_env.__class__.__name__ + "/"
        super().__init__(abstract_env, trainee, save_dir)
        self.antagonist = deepcopy(trainee)
        self.random_z_space = 10
        
        if type(self.teacher_obs_space) != dict:
            self.teacher_obs_space = dict()
            self.teacher_obs_space['data'] = abstract_env.get_observation_space()
        self.teacher_obs_space['random_z'] = self.random_z_space
        self.teacher = teacher_agent
        self.max_episode_steps = abstract_env.get_max_episode_steps()

        
    def get_next_env_block(self, obs):
        random_z  = torch.rand(self.random_z_space, device=self.device)
        obs['random_z'] = random_z
        # or act
        teacher_action = self.teacher.act(obs)
        return teacher_action


    def create_envs(self, number_of_envs=1, teacher_eval_mode=False):
    # obs = self.abstract_env.clear_env()
        self.teacher.set_num_parallel_env(number_of_envs)
        if teacher_eval_mode:
            self.teacher.set_eval_mode()
        a_env = self.abstract_env
        a_env = ParallelEnv(a_env, number_of_envs)
        if not teacher_eval_mode:
            self.teacher.set_train_mode()
            self.teacher.collect_episode_obs(a_env, max_episode_len=self.teacher_max_steps, env_funcs={"step":"step_generator", "reset":"clear_env"})
        else:
            for i in range(self.teacher_max_steps):
                obs = a_env.clear_env()
                a = self.teacher.act(obs)
                a_env.step_generator(a)
        # normal gym env again
        return a_env.get_envs()

        
    def save_models(self, additional_info):
        self.trainee.save_agent(f"{self.save_dir}_{additional_info}_trainee.ckpt")
        self.antagonist.save_agent(f"{self.save_dir}_{additional_info}_antagonist.ckpt")
        self.teacher.save_agent(f"{self.save_dir}_{additional_info}_teacher.ckpt")
        

    def set_agents_to_train_mode(self):
        self.teacher.set_train_mode()
        self.trainee.set_train_mode()
        self.antagonist.set_train_mode()


    def teach(self, n_iters, n_episodes=8):
        self.set_agents_to_train_mode()
        number_episodes_for_regret_calc = 4 # same as paired paper
        self.trainee.set_num_parallel_env(number_episodes_for_regret_calc)
        self.antagonist.set_num_parallel_env(number_episodes_for_regret_calc)
        all_mean_rewards = []
        pbar = tqdm(range(n_iters))
        number_of_envs_to_gen = 1

        for iter in pbar:
            envs = self.create_envs(number_of_envs_to_gen, teacher_eval_mode=False)
            # in paired we create single env
            env = envs[0]
            self.write_env(env, iter)
            env = ParallelEnv(env, number_episodes_for_regret_calc)
            n_steps_collected = 0
            mean_r = 0
            for i in range(n_episodes):
                trainee_rewards = self.trainee.collect_episode_obs(env, max_episode_len=self.max_episode_steps, num_to_collect=number_episodes_for_regret_calc) #collect a single episode experience - controled in num_env_parallel in each agent
                antagonist_rewards = self.antagonist.collect_episode_obs(env, max_episode_len=self.max_episode_steps, num_to_collect=number_episodes_for_regret_calc)
                episodes_collected = len(trainee_rewards)

                curr_steps_collected = np.sum([len(r) for r in trainee_rewards])

                trainee_rewards = functools.reduce(operator.iconcat, trainee_rewards, [])
                antagonist_rewards = functools.reduce(operator.iconcat, antagonist_rewards, [])

                trainee_avg_r = np.mean(trainee_rewards)
                trainee_max_r = np.max(trainee_rewards)
                anta_avg_r = np.mean(antagonist_rewards)
                anta_max_r = np.max(antagonist_rewards)

                #Change agents update... as paired paper states..
                # # update rewards:
                reward_buffer_index = self.trainee.experience.reward_index

                #sample last experiences
                t_exp = self.trainee.get_last_collected_experiences(number_episodes_for_regret_calc) # maybe change to 1 on ppo?
                a_exp = self.antagonist.get_last_collected_experiences(number_episodes_for_regret_calc) # maybe change to 1 on ppo?                

                t_exp[reward_buffer_index] = t_exp[reward_buffer_index] - (anta_max_r) / self.max_episode_steps
                a_exp[reward_buffer_index] = a_exp[reward_buffer_index] - (trainee_max_r) / self.max_episode_steps

                self.trainee.update_policy(*t_exp)
                self.antagonist.update_policy(*a_exp)

                n_steps_collected += curr_steps_collected
                mean_r +=trainee_avg_r
            
            all_mean_rewards.append(mean_r/n_episodes)
            desciption = f"R:{np.round(mean_r/n_episodes, 2):08}"
            pbar.set_description(desciption)
            # train teacher_model
            teacher_reward = (anta_max_r / self.max_episode_steps)- trainee_avg_r
            teacher_exp = self.teacher.get_last_collected_experiences(number_of_envs_to_gen) 
            teacher_exp[reward_buffer_index][-1] = teacher_reward
            self.teacher.update_policy(*teacher_exp)
            self.teacher.clear_exp()
            self.trainee.clear_exp()
            self.antagonist.clear_exp()
            if iter % self.save_agent_iters == self.save_agent_iters - 1:
                self.save_models(iter)

        return all_mean_rewards
        