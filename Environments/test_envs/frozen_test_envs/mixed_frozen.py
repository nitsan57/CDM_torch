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

  Can specify agent and goal position, if not it is set at mixed.
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
    self.easy_mixed_env_v0 = init_for_test([0,20, 9], n_clutter=1)
    self.easy_mixed_env_v0.__name__ = "easy_mixed_env_v0"

    self.easy_mixed_env_v1 = init_for_test([0,1, 5], n_clutter=1)
    self.easy_mixed_env_v1.__name__ = "easy_mixed_env_v1"

    self.easy_mixed_env_v2 = init_for_test([24,2, 5], n_clutter=1)
    self.easy_mixed_env_v2.__name__ = "easy_mixed_env_v2"

    self.easy_mixed_env_v3 = init_for_test([20,11, 1], n_clutter=1)
    self.easy_mixed_env_v3.__name__ = "easy_mixed_env_v3"

    self.easy_mixed_env_v4 = init_for_test([15,10, 20], n_clutter=1)
    self.easy_mixed_env_v4.__name__ = "easy_mixed_env_v4"

    self.easy_mixed_env_v5 = init_for_test([9,2], n_clutter=1)
    self.easy_mixed_env_v5.__name__ = "easy_mixed_env_v5"
    # mddium

    self.medium_mixed_env_v0 = init_for_test([24,9, 20, 16], n_clutter=2)
    self.medium_mixed_env_v0.__name__ = "medium_mixed_env_v0"

    self.medium_mixed_env_v1 = init_for_test([20,10, 16], n_clutter=2)
    self.medium_mixed_env_v1.__name__ = "medium_mixed_env_v1"

    self.medium_mixed_env_v2 = init_for_test([5,11, 4, 19], n_clutter=2)
    self.medium_mixed_env_v2.__name__ = "medium_mixed_env_v2"

    self.medium_mixed_env_v3 = init_for_test([19,10, 9, 20], n_clutter=2)
    self.medium_mixed_env_v3.__name__ = "medium_mixed_env_v3"

    self.medim_mixed_env_v4 = init_for_test([24,4, 10, 11], n_clutter=2)
    self.medim_mixed_env_v4.__name__ = "medim_mixed_env_v4"

    self.medium_mixed_env_v5 = init_for_test([20,2, 18,0], n_clutter=2)
    self.medium_mixed_env_v5.__name__ = "medium_mixed_env_v5"
    #Hard

    self.hard_mixed_env_v0 = init_for_test([1,9, 3, 11, 10, 12, 14], n_clutter=4)
    self.hard_mixed_env_v0.__name__ = "hard_mixed_env_v0"

    self.hard_mixed_env_v1 = init_for_test([24,0,10], n_clutter=4)
    self.hard_mixed_env_v1.__name__ = "hard_mixed_env_v1"

    self.hard_mixed_env_v2 = init_for_test([20,4,15,3], n_clutter=4)
    self.hard_mixed_env_v2.__name__ = "hard_mixed_env_v2"

    self.hard_mixed_env_v3 = init_for_test([22,1, 0,15,12], n_clutter=4)
    self.hard_mixed_env_v3.__name__ = "hard_mixed_env_v3"


    self.hard_mixed_env_v4 = init_for_test([4,10, 3, 9, 16,20], n_clutter=4)
    self.hard_mixed_env_v4.__name__ = "hard_mixed_env_v4"

    self.hard_mixed_env_v5 = init_for_test([10,20, 18,15,14], n_clutter=4)
    self.hard_mixed_env_v5.__name__ = "hard_mixed_env_v5"

    self.hard_mixed_env_v6 = init_for_test([21,0, 5,6], n_clutter=4)
    self.hard_mixed_env_v6.__name__ = "hard_mixed_env_v6"

    self.hard_mixed_env_v7 = init_for_test([22,1,4,6,12], n_clutter=4)
    self.hard_mixed_env_v7.__name__ = "hard_mixed_env_v7"

    

all_frozen_envs = GenericFrozen()

register_test_env(all_frozen_envs.easy_mixed_env_v0, "Frozen", "easy")
register_test_env(all_frozen_envs.easy_mixed_env_v1, "Frozen", "easy")
register_test_env(all_frozen_envs.easy_mixed_env_v2, "Frozen", "easy")
register_test_env(all_frozen_envs.easy_mixed_env_v3, "Frozen", "easy")
register_test_env(all_frozen_envs.easy_mixed_env_v4, "Frozen", "easy")
register_test_env(all_frozen_envs.easy_mixed_env_v5, "Frozen", "easy")

register_test_env(all_frozen_envs.medium_mixed_env_v0, "Frozen", "medium")
register_test_env(all_frozen_envs.medium_mixed_env_v1, "Frozen", "medium")
register_test_env(all_frozen_envs.medium_mixed_env_v2, "Frozen", "medium")
register_test_env(all_frozen_envs.medium_mixed_env_v3, "Frozen", "medium")
register_test_env(all_frozen_envs.medim_mixed_env_v4, "Frozen", "medium")
register_test_env(all_frozen_envs.medium_mixed_env_v5, "Frozen", "medium")

register_test_env(all_frozen_envs.hard_mixed_env_v0, "Frozen", "hard")
register_test_env(all_frozen_envs.hard_mixed_env_v1, "Frozen", "hard")
register_test_env(all_frozen_envs.hard_mixed_env_v2, "Frozen", "hard")
register_test_env(all_frozen_envs.hard_mixed_env_v3, "Frozen", "hard")
register_test_env(all_frozen_envs.hard_mixed_env_v4, "Frozen", "hard")
register_test_env(all_frozen_envs.hard_mixed_env_v5, "Frozen", "hard")
register_test_env(all_frozen_envs.hard_mixed_env_v6, "Frozen", "hard")
register_test_env(all_frozen_envs.hard_mixed_env_v7, "Frozen", "hard")







