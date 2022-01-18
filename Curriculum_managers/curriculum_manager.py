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


    @abstractmethod
    def create_envs(self, number_of_envs=1, teacher_eval_mode=False):
        raise NotImplementedError


    @abstractmethod
    def teach(self):
        raise NotImplementedError


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