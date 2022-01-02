from abc import ABC, abstractmethod
from copy import deepcopy
import os
import cv2 

from numpy.lib.npyio import save


class Curriculum_Manager(ABC):

    def __init__(self, abstract_env, trainee, save_dir) -> None:
        super().__init__()
        self.abstract_env = deepcopy(abstract_env)
        self.trainee = trainee
        self.teacher_obs_space = abstract_env.get_generator_observation_space()
        self.teacher_action_dim = abstract_env.get_generator_action_dim()
        self.teacher_max_steps = abstract_env.get_generator_max_steps()
        self.device = self.trainee.device
        self.save_agent_iters = 100
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        self.save_dir = save_dir
        self.verbose=False

    def set_verbose(self, verbose : bool):
        self.verbose = verbose


    @abstractmethod
    def create_envs(self, number_of_envs=1, teacher_eval_mode=False):
        raise NotImplementedError


    @abstractmethod
    def teach(self):
        raise NotImplementedError


    @abstractmethod
    def save_models(self, addional_info):
        raise NotImplementedError
    

    def write_env(self, env, iter):
        env.reset()
        filename = self.save_dir + "/images/" + f"{iter}.png"
        img = cv2.cvtColor(env.render('rgb_array'), cv2.COLOR_BGR2RGB)
        cv2.imwrite(filename, img)
