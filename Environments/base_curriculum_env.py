from abc import ABC, abstractmethod



class Base_Env(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_max_episode_steps(self,):
        raise NotImplementedError

    @abstractmethod
    def get_generator_max_steps(self,):

        raise NotImplementedError

    @abstractmethod
    def get_observation(self, agent=True):
        raise NotImplementedError


    @abstractmethod
    def step_generator(self, loc):
        raise NotImplementedError


    @abstractmethod
    def get_observation_space(self):
        raise NotImplementedError

    @abstractmethod
    def get_generator_observation_space(self):
        raise NotImplementedError

    @abstractmethod
    def get_action_space(self):
        raise NotImplementedError   
        
    @abstractmethod
    def get_action_dim(self):
        raise NotImplementedError   

    @abstractmethod
    def get_generator_action_space(self):
        raise NotImplementedError

    @abstractmethod
    def get_generator_action_dim(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self,):
        raise NotImplementedError


    @abstractmethod
    def clear_env(self,):
        raise NotImplementedError


    @abstractmethod
    def sample_random_state(self,):
        raise NotImplementedError


    @abstractmethod
    def step(self, actions):
        raise NotImplementedError


    @abstractmethod
    def render(self,):
        raise NotImplementedError