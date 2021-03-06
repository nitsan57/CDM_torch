from .curriculum_manager import Curriculum_Manager
import numpy as np
from copy import deepcopy
import os
from Agents.agent_utils import ParallelEnv
from tqdm import tqdm



class Random_Curriculum(Curriculum_Manager):

    def __init__(self, abstract_env, trainee, save_dir=None) -> None:
        
        super().__init__(abstract_env, trainee, save_dir)
        self.trainee = trainee
        self.max_episode_steps = abstract_env.get_max_episode_steps()


    def save_models(self, num_iter):
        self.trainee.save_agent(f'{self.save_dir}/_{num_iter}_trainee.ckpt')

    
    def load_models(self, num_iter):
        path = f'{self.save_dir}/_{num_iter}_trainee.ckpt'
        self.trainee.load_agent(path)
        return {'trainee': self.trainee}

    def create_envs(self, number_of_envs=1, teacher_eval_mode=False):
        a_env = self.abstract_env
        if number_of_envs > 1:
            a_env = ParallelEnv(a_env, number_of_envs)
        a_env.clear_env()
        # self.teacher_obs_space
        num_steps = self.teacher_max_steps

        for i in range(num_steps): # 6 stesps
            step = np.random.randint(self.teacher_action_dim)
            a_env.step_generator(step) # check action dim
        a_env.reset()
        return [a_env]


    def teach(self, n_iters, n_episodes=1):
        self.trainee.set_train_mode()
        self.trainee.set_store_entropy(True)
        number_episodes_for_regret_calc = 4
        self.trainee.set_num_parallel_env(number_episodes_for_regret_calc)
        pbar = tqdm(range(self.curr_iter, n_iters))
        for itr in pbar:
            env = self.create_envs()[0] 
            self.write_env(env, itr)

            rewards = self.trainee.train_episodial(env, n_episodes*number_episodes_for_regret_calc, max_episode_len=self.max_episode_steps, disable_tqdm=True) #train n_episodes per generated_env
            r_mean = np.mean(rewards)
            self.agent_train_rewards.append(r_mean)            
            entropy = self.get_trainne_entropy()
            self.agent_train_entropy.append(entropy)
            desciption = f"R:{np.round(np.mean(self.agent_train_rewards[-20:]), 2):08}, entropy: {entropy :01.4}"
            pbar.set_description(desciption)

            self.train_epoch_end_callbacks(itr)

                
        self.trainee.close_env_procs()
        return self.agent_train_rewards
