import imp
from .curriculum_manager import Curriculum_Manager
from copy import deepcopy
import torch
import os
import numpy as np
from Agents.agent_utils import ParallelEnv
from tqdm import tqdm
import functools
import operator
from scipy.stats import entropy as calc_entropy


class PAIRED_Curriculum_History_filter_Entropy(Curriculum_Manager):
    def __init__(self, abstract_env, trainee, teacher_agent, save_dir=None) -> None:

        self.random_z_dim = (10,)
        self.teacher = teacher_agent
        self.teacher.add_to_obs_shape({'random_z': self.random_z_dim})
        self.antagonist = deepcopy(trainee)
        self.history_env_list = []
        super().__init__(abstract_env, trainee, save_dir)
        

    def add_to_history(self, new_env_actions):
        env_dist = self.calc_env_normilized_dist(new_env_actions)
        if env_dist != 0:
            self.history_env_list.append(new_env_actions)

        
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
        all_mean_rewards = []
        pbar = tqdm(range(self.curr_iter, n_iters))
        number_of_envs_to_gen = 20
        entropy_coeff = 0.5
        history_coeff = 0.5
        max_possible_entropy = calc_entropy(np.ones(self.trainee.n_actions)/self.trainee.n_actions)
        for itr in pbar:
            envs = self.create_envs(number_of_envs_to_gen, teacher_eval_mode=False)
            teacher_exp = self.teacher.get_last_collected_experiences(number_of_envs_to_gen)
            action_buffer_index = self.trainee.experience.actions_index
            env_actions = teacher_exp[action_buffer_index]
            
            env_action_representations = [env_actions[self.teacher_max_steps*j:self.teacher_max_steps*(j+1)].detach().cpu().numpy() for j in range(number_of_envs_to_gen-1)]

            env_scores = self.score_envs(env_action_representations, self.calc_env_normilized_dist)
            env_idx = self.chose_best_env_idx(env_scores, "history")
            env = envs[env_idx]
            env_score = env_scores[env_idx]
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
            all_mean_rewards.append(total_mean_r)
            entropy = self.get_trainne_entropy()
            self.agent_train_entropy.append(entropy)
            desciption = f"R:{np.round(total_mean_r, 2):08}, entropy: {entropy :01.4}"
            pbar.set_description(desciption)

            normilized_entropy = entropy / max_possible_entropy # 1 represnts agent is NOT sure of its move 0 - sure             

            teacher_reward = (total_anta_max_r / self.max_episode_steps)- total_mean_r

            if teacher_reward > 0:
                teacher_reward+= env_score*history_coeff #the more diverse - the better if the environment is feasible
                teacher_reward +=normilized_entropy*entropy_coeff #incentivize entropy enducing env


            chosen_env_idx = (self.teacher_max_steps)*env_idx-1 #in respect to actions-rewards-buffer
            teacher_exp[reward_buffer_index][chosen_env_idx] = teacher_reward
            self.teacher.update_policy(*teacher_exp)
            self.teacher.clear_exp()
            self.trainee.clear_exp()
            self.antagonist.clear_exp()
            
            self.curr_iter = itr
            if itr % self.save_agent_iters == self.near_save_coeff:
                self.save_ckpts(itr, {"agent_train_entropy" : self.agent_train_entropy, "history_env_list": self.history_env_list})
            

        self.trainee.close_env_procs()
        self.antagonist.close_env_procs()
        return all_mean_rewards
        