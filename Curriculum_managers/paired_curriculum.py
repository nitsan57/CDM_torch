from .curriculum_manager import Curriculum_Manager
from copy import deepcopy

class PAIRED_Curriculum(Curriculum_Manager):
    def __init__(self, trainee, n_episodes = 100, n_iters=50000) -> None:
        self.antagonist = deepcopy(trainee)
        super().__init__(trainee, n_episodes, n_iters)

    def create_env(self, abstract_env):
        return super().create_env(abstract_env)

    def teach():
        pass