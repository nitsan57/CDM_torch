from abc import ABC, abstractmethod


class Curriculum_Manager(ABC):

    def __init__(self, trainee) -> None:
        super().__init__()
        self.trainee = trainee

    @abstractmethod
    def create_env(self, abstract_env):
        raise NotImplementedError

    @abstractmethod
    def teach():
        raise NotImplementedError

