from pickle import DUP

from .curriculum_manager import Curriculum_Manager
from copy import deepcopy
import torch
import os
import numpy as np
from Agents.agent_utils import ParallelEnv
from tqdm import tqdm
import functools
import operator

class Entropy_Curriculum(Curriculum_Manager):
    def __init__(self, abstract_env, trainee, teacher_agent, n_episodes = 100, n_iters=50000) -> None:
        save_dir = "./ext_paired_agent_weights/" + abstract_env.__class__.__name__ + "/"
        super().__init__(abstract_env, trainee, save_dir)
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
        # self.abstract_env.clear_env()
        self.teacher.set_num_parallel_env(number_of_envs)
        if teacher_eval_mode:
            self.teacher.set_eval_mode()
        a_env = self.abstract_env
        # if number_of_envs > 1:
            # a_env = ParallelEnv(a_env)
        import pdb
        pdb.set_trace()
        a_env = ParallelEnv(a_env, number_of_envs)
        if not teacher_eval_mode:
            self.teacher.set_train_mode()
            self.teacher.collect_episode_obs(a_env, max_episode_len=self.teacher_max_steps, env_funcs={"step":"step_generator", "reset":"clear_env"})
        else:
            obs = a_env.clear_env()
            for i in range(self.teacher_max_steps):
                a = self.teacher.act(obs, number_of_envs)
                obs, _, _, _ = a_env.step_generator(a)
        # normal gym env again
        return a_env.get_envs()

        
    def save_models(self, additional_info):
        self.trainee.save_agent(f"{self.save_dir}_{additional_info}_trainee.ckpt")
        self.teacher.save_agent(f"{self.save_dir}_{additional_info}_teacher.ckpt")
        

    def set_agents_to_train_mode(self):
        self.teacher.set_train_mode()
        self.trainee.set_train_mode()    

    def get_best_env(self, env_list):
        number_states_to_sample = 20
        entropy_list = np.zeros(len(env_list))
        for i,e in enumerate(env_list):
            all_obs = []
            for j in range(number_states_to_sample):
                s_obs = e.sample_random_state()
                all_obs.append(s_obs)

            entropy = self.trainee.get_entropy(all_obs, number_states_to_sample)
            entropy_list[i] = np.mean(entropy)

        median_idx = np.argsort(entropy_list)[len(entropy_list)//2]

        return env_list[median_idx]


    def teach(self, n_iters, n_episodes=8):
        self.set_agents_to_train_mode()
        number_episodes_for_regret_calc = 1
        self.trainee.set_num_parallel_env(number_episodes_for_regret_calc)
        all_mean_rewards = []
        pbar = tqdm(range(n_iters))
        number_of_envs_to_gen = 20

        for iter in pbar:
            envs = self.create_envs(number_of_envs_to_gen, teacher_eval_mode=False)
            env = self.get_best_env(envs)
            self.write_env(env, iter)

            env = ParallelEnv(env, number_episodes_for_regret_calc)
            n_steps_collected = 0
            mean_r = 0
            trainee_rewards = self.trainee.train_episodial(env, n_episodes, disable_tqdm=True) #train n_episodes per generated_env

            trainee_rewards = functools.reduce(operator.iconcat, trainee_rewards, [])


            trainee_avg_r = np.mean(trainee_rewards)
            trainee_max_r = np.max(trainee_rewards)
            all_mean_rewards.append(trainee_avg_r)

            desciption = f"R:{np.round(mean_r/n_episodes, 2):08}"
            pbar.set_description(desciption)

            # train teacher_model
            reward_buffer_index = self.teacher.experience.reward_index
            #TODO
            teacher_reward = -trainee_avg_r - 0.01* 50000
            teacher_exp = self.teacher.get_last_collected_experiences(number_of_envs_to_gen) 
            teacher_exp[reward_buffer_index][-1] = teacher_reward
            self.teacher.update_policy(*teacher_exp)
            self.teacher.clear_exp()
            self.antagonist.clear_exp()
            self.save_models(iter)

        return all_mean_rewards