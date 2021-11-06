from .curriculum_manager import Curriculum_Manager
import numpy as np
from copy import deepcopy
import os

class Random_Curriculum(Curriculum_Manager):

    def __init__(self, abstract_env, trainee) -> None:
        super().__init__(trainee)
        self.trainee = trainee
        self.abstract_env = deepcopy(abstract_env)

    def create_env(self):
        a_env = self.abstract_env
        a_env.clear_env()

        action_dim = a_env.adversary_action_dim
        num_steps = a_env.adversary_max_steps

        for i in range(6): # 6 stesps
            step = np.random.randint(action_dim)
            a_env.step_adversary(step) # check action dim
        a_env.reset()
        return a_env


    def teach(self, n_iters, n_steps_p_env=10000):
        save_dir = "./agent_weights"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        for i in range(n_iters):
            env = self.create_env()
            self.trainee.train_n_steps(env,n_steps_p_env) #train 10K steps per generated_env
            name = self.trainee.__class__.__name__
            self.trainee.save_agent(f'{save_dir}/{name}_{i}.ckpt')
        
