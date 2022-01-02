from .curriculum_manager import Curriculum_Manager
import numpy as np
from copy import deepcopy
import os
from Agents.agent_utils import ParallelEnv
from tqdm import tqdm


class Random_Curriculum(Curriculum_Manager):

    def __init__(self, abstract_env, trainee) -> None:
        save_dir = "./random_agent_weights/" + abstract_env.__class__.__name__ + "/"
        super().__init__(abstract_env, trainee, save_dir)
        self.trainee = trainee
        self.max_episode_steps = abstract_env.get_max_episode_steps()


    def save_models(self, addional_info):
        self.trainee.save_agent(f'{self.save_dir}/_{addional_info}_trainee.ckpt')


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
        self.trainee.set_num_parallel_env(1)
        pbar = tqdm(range(n_iters))
        all_mean_rewards = []
        for i in pbar:
            # create single rand env
            env = self.create_envs()[0] 
            self.write_env(env, iter)
            rewards = self.trainee.train_episodial(env, n_episodes, max_episode_len=self.max_episode_steps, disable_tqdm=True) #train n_episodes per generated_env
            r_mean = np.mean(rewards)
            all_mean_rewards.append(r_mean)
            desciption = f"R:{np.round(r_mean, 2):08}"
            pbar.set_description(desciption)
            if i % self.save_agent_iters == self.save_agent_iters - 1:
                self.save_models(i)

        return all_mean_rewards
