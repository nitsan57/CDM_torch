from .plr_utils import PLR_REPAIRED
from .curriculum_manager import Curriculum_Manager
from copy import deepcopy
import numpy as np
from Agents.agent_utils import ParallelEnv
from tqdm import tqdm
import copy
import pickle


class REPAIRED_Curriculum(Curriculum_Manager):
    """https://arxiv.org/pdf/2110.02439.pdf Paired + PLR"""
    def __init__(self, abstract_env, trainee, teacher_agent, save_dir=None, plr_size=4000) -> None:

        self.random_z_dim = (10,)
        self.teacher = teacher_agent
        self.teacher.add_to_obs_shape({'random_z': self.random_z_dim})
        self.antagonist = deepcopy(trainee)
        max_gen_steps = abstract_env.get_generator_max_steps()
        self.plr_pro = PLR_REPAIRED(plr_size, max_gen_steps)
        self.plr_anta = PLR_REPAIRED(plr_size,max_gen_steps)
        super().__init__(abstract_env, trainee, save_dir)


    def score(self, policy_estimated_values, R_max):
        return np.abs(R_max - np.sum(policy_estimated_values))/len(policy_estimated_values)


    def update_plr(self, agnet_id, level, score):
        if agnet_id == 'antagonist':
            self.plr_anta.add_level(level, score)
        elif agnet_id == 'protagonist':
            self.plr_anta.add_level(level, score)
        else:
            raise Exception("Not a valid arg to plr update expected one of: protagonist, antagonist")


    def create_envs(self, load_from_mem, number_of_envs=1, teacher_eval_mode=False,):
        assert number_of_envs == 1, "current not support more for this generator"
        if not load_from_mem:
            # sample from generator
            assert number_of_envs == 1 , "current more is not supported"
            self.teacher.set_num_parallel_env(number_of_envs)
            a_env = self.abstract_env
            a_env = ParallelEnv(a_env, number_of_envs)
            if not teacher_eval_mode:
                self.teacher.set_train_mode()
                random_z  = np.random.rand(self.random_z_dim[0])
                additional_const_features = {'random_z':  random_z}
                self.teacher.collect_episode_obs(a_env, max_episode_len=self.teacher_max_steps, env_funcs={"step":"step_generator", "reset":"clear_env"}, additional_const_features=additional_const_features)
            else:
                self.teacher.set_eval_mode()
                for i in range(self.teacher_max_steps):
                    obs = a_env.clear_env()
                    a = self.teacher.act(obs)
                    a_env.step_generator(a)
            self.teacher.set_train_mode()
            # normal gym env again
            # only because of const seed
            self.abstract_env = a_env.get_envs()[0]
            env = a_env.get_envs()[0]
            return env, copy.deepcopy(env)
        else:
            # sample from history
            env_params_pro=  self.plr_pro.sample_level()
            env_params_anta=  self.plr_anta.sample_level()
            env_pro = copy.deepcopy(self.abstract_env)
            env_pro.init_from_vec(env_params_pro)
            self.abstract_env.init_from_vec(env_params_anta)
            env_anta = self.abstract_env

        return env_pro, env_anta

        
    def save_models(self, num_iter):
        self.trainee.save_agent(f"{self.save_dir}_{num_iter}_trainee.ckpt")
        self.antagonist.save_agent(f"{self.save_dir}_{num_iter}_antagonist.ckpt")
        self.teacher.save_agent(f"{self.save_dir}_{num_iter}_teacher.ckpt")

        with open(f"{self.save_dir}_{num_iter}_plr_buffer.pickle", 'wb') as f:
            obj = {'plr_ants': self.plr_anta, 'plr_pro': self.plr_pro}
            pickle.dump(obj, f)
        
    
    
    def load_models(self, num_iter):
        a_path = f'{self.save_dir}_{num_iter}_trainee.ckpt'
        anta_path = f'{self.save_dir}_{num_iter}_antagonist.ckpt'
        t_path = f'{self.save_dir}_{num_iter}_teacher.ckpt'
        self.trainee.load_agent(a_path)
        self.antagonist.load_agent(anta_path)
        self.teacher.load_agent(t_path)
        with open(f"{self.save_dir}_{num_iter}_plr_buffer.pickle", 'rb') as f:
            obj = pickle.load(obj, f)
            self.plr_anta = obj['plr_anta']
            self.plr_pro = obj['plr_pro']
            
        return {'trainee': self.trainee, 'antagonist': self.antagonist}


    def set_agents_to_train_mode(self):
        self.teacher.set_train_mode()
        self.trainee.set_train_mode()
        self.antagonist.set_train_mode()
        self.trainee.set_store_entropy(True)
        self.trainee.set_store_values(True)
        self.antagonist.set_store_values(True)


    def teach(self, n_iters, n_episodes=8):
        self.set_agents_to_train_mode()
        number_episodes_for_regret_calc = 4 # same as paired paper
        self.trainee.set_num_parallel_env(number_episodes_for_regret_calc)
        self.antagonist.set_num_parallel_env(number_episodes_for_regret_calc)
        pbar = tqdm(range(self.curr_iter, n_iters))
        number_of_envs_to_gen = 1

        for itr in pbar:
            load_from_mem = np.random.rand() > 0.5 # follow replay-decision Bernoulli with d=0.5, algorithm as explained in the paper
            # if generated new env - envs[0] == envs[1] Trainee, Antagonist
            envs = self.create_envs(load_from_mem, number_of_envs_to_gen, teacher_eval_mode=False)
            trainee_env_params = envs[0].get_param_vec()
            anta_env_params = envs[1].get_param_vec()
            
            if not load_from_mem:
                # save only generated envs
                self.write_env(envs[0], itr)

            trainee_mean_r = 0
            total_anta_max_r = 0
        
            if load_from_mem:
                trainee_rewards = self.trainee.train_episodial(envs[0], n_episodes*number_episodes_for_regret_calc, disable_tqdm=True) #train n_episodes per generated_env
                antagonist_rewards = self.antagonist.train_episodial(envs[1], n_episodes*number_episodes_for_regret_calc, disable_tqdm=True) #train n_episodes per generated_env
                trainee_max_r = np.max(trainee_rewards)    
                anta_max_r = np.max(antagonist_rewards)
            else:
                for i in range(n_episodes):
                    trainee_rewards = self.trainee.collect_episode_obs(envs[0], max_episode_len=self.max_episode_steps, num_to_collect=number_episodes_for_regret_calc) #collect a single episode experience - controled in num_env_parallel in each agent
                    antagonist_rewards = self.antagonist.collect_episode_obs(envs[1], max_episode_len=self.max_episode_steps, num_to_collect=number_episodes_for_regret_calc)

                    trainee_rewards = [np.sum(r) for r in trainee_rewards]
                    antagonist_rewards = [np.sum(r) for r in antagonist_rewards]

                    trainee_avg_r = np.mean(trainee_rewards)
                    trainee_max_r = np.max(trainee_rewards)
                    # anta_avg_r = np.mean(antagonist_rewards)
                    anta_max_r = np.max(antagonist_rewards)
                    total_anta_max_r = max(anta_max_r, total_anta_max_r)

                    reward_buffer_index = self.trainee.experience.reward_index
                    # run agents on envs:
                    t_exp = self.trainee.get_last_collected_experiences(number_episodes_for_regret_calc)
                    a_exp = self.antagonist.get_last_collected_experiences(number_episodes_for_regret_calc)
                    self.agent_train_rewards.append(trainee_avg_r)
                    entropy = self.get_trainne_entropy()
                    self.agent_train_entropy.append(entropy)
                    desciption = f"R:{np.round(trainee_avg_r, 2):08}, entropy: {entropy :01.4}"
                    pbar.set_description(desciption)

                    # total_mean_r = trainee_mean_r/n_episodes
                    teacher_reward = (total_anta_max_r / self.max_episode_steps)- trainee_avg_r
                    teacher_exp = self.teacher.get_last_collected_experiences(number_of_envs_to_gen)
                    teacher_exp[reward_buffer_index][-1] = teacher_reward
                    self.teacher.update_policy(*teacher_exp)

            trainne_estimated_vals = self.get_agent_values(self.trainee)
            anta_estimated_vals = self.get_agent_values(self.antagonist)
            score_pro = self.score(trainne_estimated_vals, trainee_max_r)
            score_anta = self.score(anta_estimated_vals, anta_max_r)
            self.update_plr('protagonist',trainee_env_params,score_pro)
            self.update_plr('antagonist',anta_env_params, score_anta)
            self.teacher.clear_exp()
            self.trainee.clear_exp()
            self.antagonist.clear_exp()
            self.train_epoch_end_callbacks(itr, {'plr_anta': self.plr_anta, 'plr_pro': self.plr_anta})

            

        self.trainee.close_env_procs()
        self.antagonist.close_env_procs()
        if load_from_mem:
            return trainee_rewards
        else:
            return self.agent_train_rewards
        


