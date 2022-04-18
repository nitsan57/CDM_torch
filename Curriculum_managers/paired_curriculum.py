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
    def __init__(self, abstract_env, trainee, teacher_agent, save_dir=None) -> None:
        # if save_dir is None:
        #     save_dir = "./results/PAIRED_Curriculum/" + abstract_env.__class__.__name__ + "/"

        self.random_z_dim = (10,)
        self.teacher = teacher_agent
        self.teacher.add_to_obs_shape({'random_z': self.random_z_dim})
        self.antagonist = deepcopy(trainee)
        super().__init__(abstract_env, trainee, save_dir)
        
            

    def create_envs(self, number_of_envs=1, teacher_eval_mode=False):
    # obs = self.abstract_env.clear_env()
        assert number_of_envs == 1 , "current more is not supported"
        self.teacher.set_num_parallel_env(number_of_envs)
        if teacher_eval_mode:
            self.teacher.set_eval_mode()
        a_env = self.abstract_env
        a_env = ParallelEnv(a_env, number_of_envs)
        if not teacher_eval_mode:
            self.teacher.set_train_mode()
            random_z  = np.random.rand(self.random_z_dim[0])
            additional_const_features = {'random_z':  random_z}
            self.teacher.collect_episode_obs(a_env, max_episode_len=self.teacher_max_steps, env_funcs={"step":"step_generator", "reset":"clear_env"}, additional_const_features=additional_const_features)
        else:
            for i in range(self.teacher_max_steps):
                obs = a_env.clear_env()
                a = self.teacher.act(obs)
                a_env.step_generator(a)
        # normal gym env again
        # only because of **** const seed
        self.abstract_env = a_env.get_envs()[0]
        return a_env.get_envs()

        
    def save_models(self, num_iter):
        self.trainee.save_agent(f"{self.save_dir}_{num_iter}_trainee.ckpt")
        self.antagonist.save_agent(f"{self.save_dir}_{num_iter}_antagonist.ckpt")
        self.teacher.save_agent(f"{self.save_dir}_{num_iter}_teacher.ckpt")
    
    
    def load_models(self, num_iter):
        a_path = f'{self.save_dir}_{num_iter}_trainee.ckpt'
        anta_path = f'{self.save_dir}_{num_iter}_antagonist.ckpt'
        t_path = f'{self.save_dir}_{num_iter}_teacher.ckpt'
        self.trainee.load_agent(a_path)
        self.antagonist.load_agent(anta_path)
        self.teacher.load_agent(t_path)
        return {'trainee': self.trainee, 'antagonist': self.antagonist}

    def set_agents_to_train_mode(self):
        self.teacher.set_train_mode()
        self.trainee.set_train_mode()
        self.antagonist.set_train_mode()
        self.trainee.set_store_entropy(True)



    def teach(self, n_iters, n_episodes=8):
        self.set_agents_to_train_mode()
        number_episodes_for_regret_calc = 4 # same as paired paper
        self.trainee.set_num_parallel_env(number_episodes_for_regret_calc)
        self.antagonist.set_num_parallel_env(number_episodes_for_regret_calc)
        pbar = tqdm(range(self.curr_iter, n_iters))
        number_of_envs_to_gen = 1

        for itr in pbar:
            envs = self.create_envs(number_of_envs_to_gen, teacher_eval_mode=False)
            # in paired we create single env
            env = envs[0]
            self.write_env(env, itr)
            # env = ParallelEnv(env, number_episodes_for_regret_calc)
            trainee_mean_r = 0
            total_anta_max_r = 0
            for i in range(n_episodes):
                trainee_rewards = self.trainee.collect_episode_obs(env, max_episode_len=self.max_episode_steps, num_to_collect=number_episodes_for_regret_calc) #collect a single episode experience - controled in num_env_parallel in each agent
                antagonist_rewards = self.antagonist.collect_episode_obs(env, max_episode_len=self.max_episode_steps, num_to_collect=number_episodes_for_regret_calc)

                trainee_rewards = [np.sum(r) for r in trainee_rewards]
                antagonist_rewards = [np.sum(r) for r in antagonist_rewards]

                trainee_avg_r = np.mean(trainee_rewards)
                trainee_max_r = np.max(trainee_rewards)
                anta_avg_r = np.mean(antagonist_rewards)
                anta_max_r = np.max(antagonist_rewards)
                total_anta_max_r = max(anta_max_r, total_anta_max_r)

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

                trainee_mean_r +=trainee_avg_r
            
            total_mean_r = trainee_mean_r/n_episodes

            self.agent_train_rewards.append(total_mean_r)
            entropy = self.get_trainne_entropy()
            self.agent_train_entropy.append(entropy)
            desciption = f"R:{np.round(total_mean_r, 2):08}, entropy: {entropy :01.4}"
            pbar.set_description(desciption)

            teacher_reward = (total_anta_max_r / self.max_episode_steps)- total_mean_r
            teacher_exp = self.teacher.get_last_collected_experiences(number_of_envs_to_gen)
            teacher_exp[reward_buffer_index][-1] = teacher_reward
            self.teacher.update_policy(*teacher_exp)
            self.teacher.clear_exp()
            self.trainee.clear_exp()
            self.antagonist.clear_exp()
            
            self.train_epoch_end_callbacks(itr)

            

        self.trainee.close_env_procs()
        self.antagonist.close_env_procs()
        return self.agent_train_rewards
        