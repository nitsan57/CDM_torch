# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Implements the multi-agent version of minigrid four rooms environments.

This environment is a classic exploration problem where the goal must be located
in one of four rooms.
"""

from Environments.frozen import FrozenLakeEnv
from Environments.environments import register_test_env
from .test_frozen_utils import init_for_test

class GenericFrozen():
  """Classic 4 rooms gridworld environment.

  Can specify agent and goal position, if not it is set at random.
  """

  def __init__(self):
    """Constructor.

    Args:
        agent_pos: An array of agent positions. Length should match n_agents.
        goal_pos: An (x,y) position.
        n_agents: The number of agents in the environment.
        grid_size: The height and width of the grid.
        agent_view_size: The width of the agent's field of view in grid squares.
        two_rooms: If True, will only build the vertical wall.
        minigrid_mode: If True, observations come back without the multi-agent
          dimension.
        **kwargs: See superclass.
    """
    self.easy_random_env_v0 = init_for_test([0,5, 9, 10], n_clutter=4)
    self.easy_random_env_v0.__name__ = "easy_random_env_v0"

    self.easy_random_env_v1 = init_for_test([20,15, 0,5], n_clutter=4)
    self.easy_random_env_v1.__name__ = "easy_random_env_v1"

    self.easy_random_env_v2 = init_for_test([6,2, 0,20], n_clutter=4)
    self.easy_random_env_v2.__name__ = "easy_random_env_v2"

    self.easy_random_env_v3 = init_for_test([0,5, 9, 0], n_clutter=4)
    self.easy_random_env_v3.__name__ = "easy_random_env_v3"

    self.easy_random_env_v4 = init_for_test([20,15, 20,15], n_clutter=4)
    self.easy_random_env_v4.__name__ = "easy_random_env_v4"

    self.easy_random_env_v5 = init_for_test([20,2], n_clutter=4)
    self.easy_random_env_v5.__name__ = "easy_random_env_v5"
    #Medium
    self.medium_random_env_v0 = init_for_test([0,7, 3, 9, 10, 12, 14], n_clutter=6)
    self.medium_random_env_v0.__name__ = "medium_random_env_v0"

    self.medium_random_env_v1 = init_for_test([0,5, 3,10,4], n_clutter=4)
    self.medium_random_env_v1.__name__ = "medium_random_env_v1"

    self.medium_random_env_v2 = init_for_test([21,0,1,3,10], n_clutter=6)
    self.medium_random_env_v2.__name__ = "medium_random_env_v2"

    self.medium_random_env_v3 = init_for_test([22,1, 0,15,13], n_clutter=6)
    self.medium_random_env_v3.__name__ = "medium_random_env_v3"


    self.medium_random_env_v4 = init_for_test([0,7, 3, 9, 10, 12, 14], n_clutter=6)
    self.medium_random_env_v4.__name__ = "medium_random_env_v4"

    self.medium_random_env_v5 = init_for_test([0,5, 0,0], n_clutter=4)
    self.medium_random_env_v5.__name__ = "medium_random_env_v5"

    self.medium_random_env_v6 = init_for_test([21,0, 0,0], n_clutter=6)
    self.medium_random_env_v6.__name__ = "medium_random_env_v6"

    self.medium_random_env_v7 = init_for_test([22,1,0,0,0], n_clutter=6)
    self.medium_random_env_v7.__name__ = "medium_random_env_v7"
    #Hard

    self.hard_random_env_v0 = init_for_test([24,0, 14,17,20,22], n_clutter=6)
    self.hard_random_env_v0.__name__ = "hard_random_env_v0"

    self.hard_random_env_v1 = init_for_test([10,24,10,11,17,22,15 ], n_clutter=7)
    self.hard_random_env_v1.__name__ = "hard_random_env_v1"

    self.hard_random_env_v2= init_for_test([5,20, 6,21,15,17,23,19], n_clutter=7)
    self.hard_random_env_v2.__name__ = "hard_random_env_v2"


    self.hard_random_env_v3 = init_for_test([0,24,1,2,12,22,17,20], n_clutter=7)
    self.hard_random_env_v3.__name__ = "hard_random_env_v3"

    self.hard_random_env_v4 = init_for_test([0,14,1,2,3,20,22,23 ], n_clutter=7)
    self.hard_random_env_v4.__name__ = "hard_random_env_v4"

    self.hard_random_env_v5= init_for_test([3,19,18,5,7,9,12], n_clutter=7)
    self.hard_random_env_v5.__name__ = "hard_random_env_v5"
    

all_frozen_envs = GenericFrozen()

register_test_env(all_frozen_envs.easy_random_env_v0, "Frozen", "easy")
register_test_env(all_frozen_envs.easy_random_env_v1, "Frozen", "easy")
register_test_env(all_frozen_envs.easy_random_env_v2, "Frozen", "easy")
register_test_env(all_frozen_envs.easy_random_env_v3, "Frozen", "easy")
register_test_env(all_frozen_envs.easy_random_env_v4, "Frozen", "easy")
register_test_env(all_frozen_envs.easy_random_env_v5, "Frozen", "easy")

register_test_env(all_frozen_envs.medium_random_env_v0, "Frozen", "medium")
register_test_env(all_frozen_envs.medium_random_env_v1, "Frozen", "medium")
register_test_env(all_frozen_envs.medium_random_env_v2, "Frozen", "medium")
register_test_env(all_frozen_envs.medium_random_env_v3, "Frozen", "medium")
register_test_env(all_frozen_envs.medium_random_env_v4, "Frozen", "medium")
register_test_env(all_frozen_envs.medium_random_env_v5, "Frozen", "medium")
register_test_env(all_frozen_envs.medium_random_env_v6, "Frozen", "medium")
register_test_env(all_frozen_envs.medium_random_env_v7, "Frozen", "medium")

register_test_env(all_frozen_envs.hard_random_env_v0, "Frozen", "hard")
register_test_env(all_frozen_envs.hard_random_env_v1, "Frozen", "hard")
register_test_env(all_frozen_envs.hard_random_env_v2, "Frozen", "hard")
register_test_env(all_frozen_envs.hard_random_env_v2, "Frozen", "hard")
register_test_env(all_frozen_envs.hard_random_env_v3, "Frozen", "hard")
register_test_env(all_frozen_envs.hard_random_env_v4, "Frozen", "hard")
register_test_env(all_frozen_envs.hard_random_env_v5, "Frozen", "hard")






