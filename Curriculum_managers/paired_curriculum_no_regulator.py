from .curriculum_manager import Curriculum_Manager
from copy import deepcopy
import torch
import os
import numpy as np
from Agents.agent_utils import ParallelEnv
from tqdm import tqdm
import functools
import operator


class Curriculum_Entropy_Only(Curriculum_Manager):
    def __init__(self, abstract_env, trainee, teacher_agent, save_dir=None) -> None:
        if save_dir is None:
            save_dir = "./results/PAIRED_Curriculum_no_regret_entropy/" + abstract_env.__class__.__name__ + "/"
        
        self.random_z_dim = (10,)
        self.teacher = teacher_agent
        self.teacher.add_to_obs_shape({'random_z': self.random_z_dim})
        super().__init__(abstract_env, trainee, save_dir)


    def create_envs(self, number_of_envs=1, teacher_eval_mode=False):
        assert number_of_envs == 1 , "current more is not supported"
        self.teacher.set_num_parallel_env(number_of_envs)
        if teacher_eval_mode:
            self.teacher.set_eval_mode()

        a_env = ParallelEnv(self.abstract_env, number_of_envs)

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
        self.abstract_env = a_env.get_envs()[0]

        return a_env.get_envs()

        
    def save_models(self, num_iter):
        self.trainee.save_agent(f"{self.save_dir}_{num_iter}_trainee.ckpt")
        self.teacher.save_agent(f"{self.save_dir}_{num_iter}_teacher.ckpt")
    
    
    def load_models(self, num_iter):
        a_path = f'{self.save_dir}_{num_iter}_trainee.ckpt'
        t_path = f'{self.save_dir}_{num_iter}_teacher.ckpt'
        self.trainee.load_agent(a_path)
        self.teacher.load_agent(t_path)
        return {'trainee': self.trainee}


    def set_agents_to_train_mode(self):
        self.teacher.set_train_mode()
        self.trainee.set_train_mode()


    def teach(self, n_iters, n_episodes=8):
        self.set_agents_to_train_mode()
        all_mean_rewards = []
        pbar = tqdm(range(self.curr_iter, n_iters))
        number_episodes_for_regret_calc = 4 # same as paired paper
        self.trainee.set_num_parallel_env(number_episodes_for_regret_calc)
        entropy_coeff = 0.001

        paired_to_calc = 4
        
        number_of_envs_to_gen = 1
        self.trainee.set_store_entropy(True)
        for itr in pbar:
            envs = self.create_envs(number_of_envs_to_gen, teacher_eval_mode=False)
            # in paired we create single env
            env = envs[0]
            self.write_env(env, itr)
            mean_r = 0

            trainee_rewards = self.trainee.train_episodial(env, n_episodes*paired_to_calc, disable_tqdm=True) #train n_episodes per generated_env

            trainee_avg_r = np.mean(trainee_rewards)

            mean_r +=trainee_avg_r
            
            all_mean_rewards.append(mean_r/n_episodes)

            entropy = self.trainee.get_stored_entropy()
            
            entropy = functools.reduce(operator.iconcat, entropy, [])
            if entropy == []:
                entropy = 1
            else:
                entropy = np.mean(entropy).astype(np.float32)


            desciption = f"R:{np.round(mean_r/n_episodes, 2):08}, entropy: {entropy :01.4}"
            pbar.set_description(desciption)

            self.trainee.clear_stored_entropy()

            teacher_reward =  trainee_avg_r - entropy

            teacher_exp = self.teacher.get_last_collected_experiences(number_of_envs_to_gen)
            reward_buffer_index = self.trainee.experience.reward_index
            teacher_exp[reward_buffer_index][-1] = teacher_reward
            self.teacher.update_policy(*teacher_exp)
            # not a must on Q learning, or any
            self.teacher.clear_exp()
            self.trainee.clear_exp()

            self.curr_iter = itr
            if itr % self.save_agent_iters == self.near_save_coeff:
                self.save_ckpts(itr)

        self.trainee.close_env_procs()
        return all_mean_rewards
        