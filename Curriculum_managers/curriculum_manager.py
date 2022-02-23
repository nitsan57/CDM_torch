from abc import ABC, abstractmethod
from copy import deepcopy
import os
import cv2 
import os
import pickle
import numpy as np 
import functools
import operator
from scipy.stats import entropy as calc_entropy
from Agents.agent_utils import ParallelEnv


class Curriculum_Manager(ABC):

    def __init__(self, abstract_env, trainee, save_dir) -> None:
        super().__init__()
        self.abstract_env = deepcopy(abstract_env)
        self.trainee = trainee
        self.teacher_obs_space = abstract_env.get_generator_observation_space()
        self.teacher_action_dim = abstract_env.get_generator_action_dim()
        self.teacher_max_steps = abstract_env.get_generator_max_steps()
        self.max_episode_steps = abstract_env.get_max_episode_steps()
        self.agent_train_entropy = []


        self.device = self.trainee.device
        self.save_agent_iters = 100
        if save_dir is None:
            save_dir = "./results/"+self.__class__.__name__ +"/"+ abstract_env.__class__.__name__ + "/"

        if not os.path.isdir(save_dir):
            os.makedirs(os.path.join(save_dir, "images"))
        self.save_dir = save_dir
        self.verbose=False
        self.meta_data_file = os.path.join(self.save_dir, 'meta_data.pkl')
        self.curr_iter = 0
        self.near_save_coeff = self.save_agent_iters - 1
        if os.path.isfile(self.meta_data_file):
            print(f'loading metadatafrom {self.meta_data_file}')
            self.load_meta_data()
            print(f'loading models from last iter: {self.curr_iter}')
            self.load_models(self.curr_iter)
        else:
            print(f'no files to load from')
            

    def get_trainne_entropy(self):
        entropy = self.trainee.get_stored_entropy()
        entropy = functools.reduce(operator.iconcat, entropy, [])
        if entropy == []:
            entropy = 1
        else:
            entropy = np.mean(entropy).astype(np.float32)
        self.trainee.clear_stored_entropy()

        return entropy

    def save_ckpts(self, i, extra_meta_data=None):
        self.save_models(i)
        # extra_meta_data["agent_train_rewards"] = self.trainee.
        self.save_meta_data(i, extra_meta_data)


    def save_meta_data(self, i, extra_data : dict=None):
        data = {'curr_iter': i}
        file = open(self.meta_data_file, 'wb')
        if extra_data is None:
            pickle.dump(data, file)
        else:
            for k in extra_data:
                # assert type(extra_data[k]) in [list], "bad type to save as extra data"
                data[k] = extra_data[k]
            pickle.dump(data, file)
        
        file.close()

    def load_meta_data(self):
        file = open(self.meta_data_file, 'rb')
        meta_data = pickle.load(file)
        file.close()
        # self.curr_iter = meta_data['curr_iter']
        for k in meta_data:
            setattr(self, k, meta_data[k])

    def set_verbose(self, verbose : bool):
        self.verbose = verbose


    def create_envs(self, number_of_envs=1, teacher_eval_mode=False):
        env_list = []
        if teacher_eval_mode:
                self.teacher.set_eval_mode()
        else:
            self.teacher.set_train_mode()

        self.teacher.set_num_parallel_env(1)
        for i in range(number_of_envs):
            a_env = self.abstract_env
            a_env = ParallelEnv(a_env, 1)

            random_z  = np.random.rand(self.random_z_dim[0])
            additional_const_features = {'random_z':  random_z}
            self.teacher.collect_episode_obs(a_env, max_episode_len=self.teacher_max_steps, env_funcs={"step":"step_generator", "reset":"clear_env"}, additional_const_features=additional_const_features)
            self.abstract_env = a_env.get_envs()[0]
            env_list.append(deepcopy(a_env.get_envs()[0]))

        return env_list


    def chose_best_env_idx(self, score_list, method):
        
        if method == "history":
            return np.argmax(score_list)

        else:
            raise Exception("not implimented method")

    def score_envs(self, env_list, scoring_func):
        env_scores = []
        for env in env_list:
            env_score = scoring_func(env)
            env_scores.append(env_score)
        return env_scores

    def calc_env_normilized_dist(self, new_env_actions):
        global_max_dist = np.sum(np.ones(self.teacher_max_steps)*self.teacher_action_dim)
        env_dist = global_max_dist
        for h_e in self.history_env_list:
            env_dist = min(np.sum(np.abs(new_env_actions - h_e), env_dist))
        return env_dist / global_max_dist

    @abstractmethod
    def teach(self):
        raise NotImplementedError

    # def teach_changed_rewards(self, n_iters, n_episodes=8):
    #     self.set_agents_to_train_mode()
    #     number_episodes_for_regret_calc = 4 # same as paired paper
    #     self.trainee.set_num_parallel_env(number_episodes_for_regret_calc)
    #     self.antagonist.set_num_parallel_env(number_episodes_for_regret_calc)
    #     all_mean_rewards = []
    #     pbar = tqdm(range(self.curr_iter, n_iters))
    #     number_of_envs_to_gen = 1

    #     for itr in pbar:
    #         envs = self.create_envs(number_of_envs_to_gen, teacher_eval_mode=False)
    #         # in paired we create single env
    #         env = envs[0]
    #         self.write_env(env, itr)
    #         # env = ParallelEnv(env, number_episodes_for_regret_calc)
    #         trainee_mean_r = 0
    #         total_anta_max_r = 0
    #         for i in range(n_episodes):
    #             trainee_rewards = self.trainee.collect_episode_obs(env, max_episode_len=self.max_episode_steps, num_to_collect=number_episodes_for_regret_calc) #collect a single episode experience - controled in num_env_parallel in each agent
    #             antagonist_rewards = self.antagonist.collect_episode_obs(env, max_episode_len=self.max_episode_steps, num_to_collect=number_episodes_for_regret_calc)

    #             trainee_rewards = [np.sum(r) for r in trainee_rewards]
    #             antagonist_rewards = [np.sum(r) for r in antagonist_rewards]

    #             trainee_avg_r = np.mean(trainee_rewards)
    #             trainee_max_r = np.max(trainee_rewards)
    #             anta_avg_r = np.mean(antagonist_rewards)
    #             anta_max_r = np.max(antagonist_rewards)
    #             total_anta_max_r = max(anta_max_r, total_anta_max_r)

    #             #Change agents update... as paired paper states..
    #             # # update rewards:
    #             reward_buffer_index = self.trainee.experience.reward_index

    #             #sample last experiences
    #             t_exp = self.trainee.get_last_collected_experiences(number_episodes_for_regret_calc) # maybe change to 1 on ppo?
    #             a_exp = self.antagonist.get_last_collected_experiences(number_episodes_for_regret_calc) # maybe change to 1 on ppo?                

    #             t_exp[reward_buffer_index] = t_exp[reward_buffer_index] - (anta_max_r) / self.max_episode_steps
    #             a_exp[reward_buffer_index] = a_exp[reward_buffer_index] - (trainee_max_r) / self.max_episode_steps

    #             self.trainee.update_policy(*t_exp)
    #             self.antagonist.update_policy(*a_exp)

    #             trainee_mean_r +=trainee_avg_r
            
    #         total_mean_r = trainee_mean_r/n_episodes
    #         all_mean_rewards.append(total_mean_r)
    #         entropy = self.get_trainne_entropy()
    #         self.agent_train_entropy.append(entropy)
    #         desciption = f"R:{np.round(total_mean_r, 2):08}, entropy: {entropy :01.4}"
    #         pbar.set_description(desciption)

    #         teacher_reward = (total_anta_max_r / self.max_episode_steps)- total_mean_r
    #         teacher_exp = self.teacher.get_last_collected_experiences(number_of_envs_to_gen)
    #         teacher_exp[reward_buffer_index][-1] = teacher_reward
    #         self.teacher.update_policy(*teacher_exp)
    #         self.teacher.clear_exp()
    #         self.trainee.clear_exp()
    #         self.antagonist.clear_exp()
            
    #         self.curr_iter = itr
    #         if itr % self.save_agent_iters == self.near_save_coeff:
    #             self.save_ckpts(itr, {"agent_train_entropy" : self.agent_train_entropy})
            

    #     self.trainee.close_env_procs()
    #     self.antagonist.close_env_procs()
    #     return all_mean_rewards



    @abstractmethod
    def save_models(self, num_iter):
        raise NotImplementedError
    
    @abstractmethod
    def load_models(self, *path_dict):
        raise NotImplementedError
    

    def write_env(self, env, iter):
        env.reset()
        filename = self.save_dir + "/images/" + f"{iter}.png"
        img = cv2.cvtColor(env.render('rgb_array'), cv2.COLOR_BGR2RGB)
        cv2.imwrite(filename, img)