from operator import concat
from .curriculum_manager import Curriculum_Manager
from copy import deepcopy
import torch
import os
import numpy as np
from Agents.agent_utils import ParallelEnv
from tqdm import tqdm
from scipy.stats import entropy as calc_entropy


class Curriculum_Entropy_History(Curriculum_Manager):
    def __init__(self, abstract_env, trainee, teacher_agent, inv_reward_entropy_coeff=1, save_dir=None) -> None:
        # if save_dir is None:
        #     save_dir = "./results/Curriculum_Entropy_History/" + abstract_env.__class__.__name__ + "/"
        
        self.random_z_dim = (10,)
        self.teacher = teacher_agent
        self.teacher.add_to_obs_shape({'random_z': self.random_z_dim})
        self.inv_reward_entropy_coeff = inv_reward_entropy_coeff
        self.max_reward = 0
        self.history_env_list = []
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
        self.trainee.set_store_entropy(True)
    

    def calc_env_normilized_dist_and_add_to_history(self, new_env_actions):
        global_max_dist = np.sum(np.ones(self.teacher_max_steps)*self.teacher_action_dim)
        env_dist = 0
        for h_e in self.history_env_list:
            env_dist = max(np.sum(np.abs(new_env_actions - h_e), env_dist))
            
        if env_dist != 0:
            self.history_env_list.append(new_env_actions)

        return env_dist / global_max_dist


    def teach(self, n_iters, n_episodes=8):
        self.set_agents_to_train_mode()
        all_mean_rewards = []
        pbar = tqdm(range(self.curr_iter, n_iters))
        number_episodes_for_regret_calc = 4 # same as paired paper
        self.trainee.set_num_parallel_env(number_episodes_for_regret_calc)

        entropy_scale = 1
        history_coeff = 2
        paired_to_calc = 4
        
        number_of_envs_to_gen = 1
        for itr in pbar:
            envs = self.create_envs(number_of_envs_to_gen, teacher_eval_mode=False)
            # in paired we create single env
            env = envs[0]
            self.write_env(env, itr)
            mean_r = 0

            trainee_rewards = self.trainee.train_episodial(env, n_episodes*paired_to_calc, disable_tqdm=True) #train n_episodes per generated_env

            trainee_avg_r = np.mean(trainee_rewards)
            trainee_max_r = np.max(trainee_rewards)
            self.max_reward = max(np.max(np.abs(trainee_rewards)), self.max_reward)

            mean_r +=trainee_avg_r
            
            all_mean_rewards.append(mean_r/n_episodes)


            entropy = self.get_trainne_entropy()
            self.agent_train_entropy.append(entropy)
            desciption = f"R:{np.round(mean_r/n_episodes, 2):08}, entropy: {entropy :01.4}"
            pbar.set_description(desciption)

            max_possible_entropy = calc_entropy(np.ones(self.trainee.n_actions)/self.trainee.n_actions)
            normilized_inv_entropy = (max_possible_entropy - entropy) / max_possible_entropy # 1 represnts agent is sure of its move (1-entorpy)
            rescaled_trainee_reward = trainee_max_r / (self.max_reward+1e-8)

            

            teacher_exp = self.teacher.get_last_collected_experiences(number_of_envs_to_gen)
            reward_buffer_index = self.trainee.experience.reward_index
            action_buffer_index = self.trainee.experience.actions_index
            env_actions = teacher_exp[action_buffer_index]
            # print("all teacher stats:", rescaled_trainee_reward, normilized_inv_entropy, self.calc_env_normilized_dist_and_add_to_history(env_actions))
            teacher_reward =  rescaled_trainee_reward - normilized_inv_entropy*self.inv_reward_entropy_coeff - (1-self.calc_env_normilized_dist_and_add_to_history(env_actions))*history_coeff
            teacher_exp[reward_buffer_index][-1] = teacher_reward
            self.teacher.update_policy(*teacher_exp)
            # not a must on Q learning, or any
            self.teacher.clear_exp()
            self.trainee.clear_exp()

            self.curr_iter = itr
            if itr % self.save_agent_iters == self.near_save_coeff:
                self.save_ckpts(itr, {"agent_train_entropy" : self.agent_train_entropy, "max_reward": self.max_reward, "history_env_list": self.history_env_list})

        self.trainee.close_env_procs()
        return all_mean_rewards
        