from .curriculum_manager import Curriculum_Manager
from copy import deepcopy
import torch
import os
import numpy as np
from Agents.agent_utils import ParallelEnv
from tqdm import tqdm
import functools
import operator


class PAIRED_Curriculum_no_regret(Curriculum_Manager):
    def __init__(self, abstract_env, trainee, teacher_agent, save_dir=None) -> None:
        if save_dir is None:
            save_dir = "./results/PAIRED_Curriculum_no_regret/" + abstract_env.__class__.__name__ + "/"

        self.antagonist = deepcopy(trainee)
        self.random_z_dim = (10,)
        self.teacher = teacher_agent
        self.teacher.add_to_obs_shape({'random_z': self.random_z_dim})
        super().__init__(abstract_env, trainee, save_dir)

        
    def get_next_env_block(self, obs):
        random_z  = torch.rand(self.random_z_dim, device=self.device)
        obs['random_z'] = random_z
        # or act
        teacher_action = self.teacher.act(obs)
        return teacher_action


    def create_envs(self, number_of_envs=1, teacher_eval_mode=False):
        assert number_of_envs == 1 , "current more is not supported"
    # obs = self.abstract_env.clear_env()
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
        

    def set_agents_to_train_mode(self):
        self.teacher.set_train_mode()
        self.trainee.set_train_mode()
        self.antagonist.set_train_mode()


    def teach(self, n_iters, n_episodes=8):
        self.set_agents_to_train_mode()
        all_mean_rewards = []
        pbar = tqdm(range(self.curr_iter, n_iters))
        number_of_envs_to_gen = 1

        for itr in pbar:
            envs = self.create_envs(number_of_envs_to_gen, teacher_eval_mode=False)
            # in paired we create single env
            env = envs[0]
            self.write_env(env, itr)
            mean_r = 0
            for i in range(n_episodes):
                trainee_rewards = self.trainee.train_episodial(env, n_episodes, disable_tqdm=True) #train n_episodes per generated_env      
                antagonist_rewards = self.antagonist.train_episodial(env, n_episodes, disable_tqdm=True) #train n_episodes per generated_env

                trainee_avg_r = np.mean(trainee_rewards)
                trainee_max_r = np.max(trainee_rewards)
                anta_avg_r = np.mean(antagonist_rewards)
                anta_max_r = np.max(antagonist_rewards)

                mean_r +=trainee_avg_r
            
            all_mean_rewards.append(mean_r/n_episodes)
            desciption = f"R:{np.round(mean_r/n_episodes, 2):08}"
            pbar.set_description(desciption)
            # train teacher_model
            if anta_max_r > trainee_max_r:
                teacher_reward = (anta_max_r / self.max_episode_steps)- trainee_avg_r
            else:
                teacher_reward = (trainee_avg_r / self.max_episode_steps)- anta_avg_r

            teacher_exp = self.teacher.get_last_collected_experiences(number_of_envs_to_gen)
            reward_buffer_index = self.trainee.experience.reward_index
            teacher_exp[reward_buffer_index][-1] = teacher_reward
            self.teacher.update_policy(*teacher_exp)
            # not a must on Q learning, or any
            self.teacher.clear_exp()
            self.trainee.clear_exp()
            self.antagonist.clear_exp()

            self.curr_iter = itr
            if itr % self.save_agent_iters == self.near_save_coeff:
                self.save_ckpts(itr)

        self.trainee.close_env_procs()
        self.antagonist.close_env_procs()
        return all_mean_rewards
        