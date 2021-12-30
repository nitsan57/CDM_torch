from abc import ABC, abstractmethod
from copy import deepcopy

class Curriculum_Manager(ABC):

    def __init__(self, abstract_env, trainee) -> None:
        super().__init__()
        self.abstract_env = deepcopy(abstract_env)
        self.trainee = trainee
        self.teacher_obs_space = abstract_env.get_generator_observation_space()
        self.teacher_action_dim = abstract_env.get_generator_action_dim()
        self.teacher_max_steps = abstract_env.get_generator_max_steps()
        self.device = self.trainee.device
        self.save_agent_iters = 100


    @abstractmethod
    def create_env(self, number_of_envs=1, teacher_eval_mode=False):
        raise NotImplementedError

    @abstractmethod
    def teach(self):
        raise NotImplementedError
