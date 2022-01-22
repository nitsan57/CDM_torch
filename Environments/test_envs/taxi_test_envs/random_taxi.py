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

from Environments.taxi import SingleTaxiEnv
from Environments.environments import register_test_env
from .test_taxi_utils import init_for_test

class GenericTaxi():
  """Classic 4 rooms gridworld environment.

  Can specify agent and goal position, if not it is set at random.
  """

  def __init__(self, size=5, agent_view_size=3, max_steps=300, n_clutter=10, n_agents=1, dist_vec =None):
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
    self.easy_random_env_v0 = init_for_test(None, True, size=5, agent_view_size=3, max_steps=300, n_clutter=0, n_agents=1, random_reset_loc=False)
    self.easy_random_env_v0.__name__ = "easy_random_env_v0"

    self.easy_random_env_v1 = init_for_test(None, True, size=5, agent_view_size=3, max_steps=300, n_clutter=0, n_agents=1, random_reset_loc=False)
    self.easy_random_env_v1.__name__ = "easy_random_env_v1"

    self.easy_random_env_v2 = init_for_test(None, True, size=5, agent_view_size=3, max_steps=300, n_clutter=0, n_agents=1, random_reset_loc=False)
    self.easy_random_env_v2.__name__ = "easy_random_env_v2"

    #Medium
    self.medium_random_env_v0 = init_for_test(None, True, size=5, agent_view_size=3, max_steps=300, n_clutter=4, n_agents=1, random_reset_loc=False)
    self.medium_random_env_v0.__name__ = "medium_random_env_v0"

    self.medium_random_env_v0 = init_for_test(None, True, size=5, agent_view_size=3, max_steps=300, n_clutter=4, n_agents=1, random_reset_loc=False)
    self.medium_random_env_v0.__name__ = "medium_random_env_v0"

    self.medium_random_env_v1 = init_for_test(None, True, size=5, agent_view_size=3, max_steps=300, n_clutter=4, n_agents=1, random_reset_loc=False)
    self.medium_random_env_v1.__name__ = "medium_random_env_v1"

    self.medium_random_env_v2 = init_for_test(None, True, size=5, agent_view_size=3, max_steps=300, n_clutter=4, n_agents=1, random_reset_loc=False)
    self.medium_random_env_v2.__name__ = "medium_random_env_v2"
    #Hard

    self.hard_random_env_v0 = init_for_test(None, True, size=5, agent_view_size=3, max_steps=300, n_clutter=12, n_agents=1, random_reset_loc=False)
    self.hard_random_env_v0.__name__ = "hard_random_env_v0"

    self.hard_random_env_v1 = init_for_test(None, True, size=5, agent_view_size=3, max_steps=300, n_clutter=12, n_agents=1, random_reset_loc=False)
    self.hard_random_env_v1.__name__ = "hard_random_env_v1"

    self.hard_random_env_v2= init_for_test(None, True, size=5, agent_view_size=3, max_steps=300, n_clutter=12, n_agents=1, random_reset_loc=False)
    self.hard_random_env_v2.__name__ = "hard_random_env_v2"
    

all_taxi_envs = GenericTaxi()

register_test_env(all_taxi_envs.easy_random_env_v0, "Taxi", "easy")
register_test_env(all_taxi_envs.easy_random_env_v1, "Taxi", "easy")
register_test_env(all_taxi_envs.easy_random_env_v2, "Taxi", "easy")
register_test_env(all_taxi_envs.medium_random_env_v0, "Taxi", "medium")
register_test_env(all_taxi_envs.medium_random_env_v1, "Taxi", "medium")
register_test_env(all_taxi_envs.medium_random_env_v2, "Taxi", "medium")
register_test_env(all_taxi_envs.hard_random_env_v0, "Taxi", "hard")
register_test_env(all_taxi_envs.hard_random_env_v1, "Taxi", "hard")
register_test_env(all_taxi_envs.hard_random_env_v2, "Taxi", "hard")




