from .curriculum_manager import Curriculum_Manager
import numpy as np
from copy import deepcopy
import os
from Agents.agent_utils import ParallelEnv
from tqdm import tqdm

class Random_Curriculum(Curriculum_Manager):

    def __init__(self, abstract_env, trainee) -> None:
        super().__init__(abstract_env, trainee)
        self.trainee = trainee


    def create_env(self, number_of_envs=1, teacher_eval_mode=False):
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
        return a_env


    def teach(self, n_iters, n_episodes=1):
        self.trainee.set_num_parallel_env(1)
        save_dir = "./random_agent_weights"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        pbar = tqdm(range(n_iters))
        all_mean_rewards = []
        for i in pbar:
            env = self.create_env()
            rewards = self.trainee.train_episodial(env, n_episodes, disable_tqdm=True) #train n_episodes per generated_env
            r_mean = np.mean(rewards)
            name = self.trainee.__class__.__name__
            desciption = f"R:{np.round(r_mean, 2):08}"
            pbar.set_description(desciption)
            # if i % self.save_agent_iters == 99:
                # self.trainee.save_agent(f'{save_dir}/{name}_{i}.ckpt')
        return all_mean_rewards