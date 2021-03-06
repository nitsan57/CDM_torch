from Environments.pddl_env_utils import get_pddl_env
from Environments.base_curriculum_env import Base_Env
from .environments import register_env
import gym
import numpy as np
import os
from pddlgym.pddlgym.core import PDDLEnv


def get_mov_str(x, y, walls):
    all_move_str = "\n"
    dirs = ["left", "up", "right", "down"]
    left_move = (x-1,y)
    up_move = (x,y-1)
    right_move = (x+1,y)
    down_move = (x,y+1)
    orig =(x,y)
    moves = [left_move, up_move, right_move, down_move]
    for i,dir in enumerate(dirs):
        wall_bool = False
        for w in walls:
            if all(moves[i] == w) or all(orig == w):
                wall_bool = True
                break

        if wall_bool:
            continue
        all_move_str += f"	(move-dir pos-{x}-{y} pos-{moves[i][0]}-{moves[i][1]} dir-{dir})" + "\n"

    return all_move_str

def get_non_goal_pos_str(x, y):
    pos_style = f"	(is-nongoal pos-{x}-{y})"
    return pos_style


def get_clear_pos_str(x, y):
    clear_pos_style = f"	(clear pos-{x}-{y})"
    return clear_pos_style


def create_env_file(f_path, player_pos, stone_pos, goal_pos, wall_pos=[], size=5):
    player_str = f"	(at player-01 pos-{player_pos[0]}-{player_pos[1]})\n"
    stone_str = f"	(at stone-01 pos-{stone_pos[0]}-{stone_pos[1]})\n"
    goal_str = f"	(is-goal pos-{goal_pos[0]}-{goal_pos[1]})\n"
    all_pos = np.array(list(np.ndindex(size, size)))+1
    all_clears_str = "\n"
    all_indices = np.array(list(np.ndindex(size+1, size+1)))
    bot = all_indices[:size+1]+1
    top = all_indices[-size-1:]+1
    right = all_indices[list(range(0, (size+1)**2, size+1)), :size+1]+1
    left = all_indices[list(range(size, (size+1)**2+size-1, size+1)), :size+1]+1
    walls = np.concatenate([bot, right, left, top])
    if len(wall_pos) != 0:
        new_locs = np.array(wall_pos)
        walls = np.concatenate([bot, right, left, top, new_locs])
    else:
        walls = np.concatenate([bot, right, left, top])
    for p in all_pos:
        walls_bool = True
        for w in walls:           
            if all(p == w):
                walls_bool = False
                break

        if any(player_pos != p) and any(stone_pos != p) and walls_bool:
            pos_str = get_clear_pos_str(*p)
            all_clears_str += pos_str + "\n"

    non_goal_str = "\n"
    for p in all_indices:
        p = p+1
        if any(p != goal_pos):
            non_goal_str += get_non_goal_pos_str(*p) + "\n"

    goal_str = f"	(is-goal pos-{goal_pos[0]}-{goal_pos[1]})\n"
    all_move_str = "\n"
    for p in all_pos:
        all_move_str+= get_mov_str(*p, walls)
    all_file_str = prefix + player_str + stone_str + goal_str + all_clears_str + non_goal_str + postfix + all_move_str + "))"
    curr_dir = os.path.dirname(__file__)
    f_path = curr_dir+"/../pddlgym/pddlgym/pddl/sokoban/problem00.pddl"
    with open(f_path, "w") as f:
        f.write(all_file_str)


class Sokoban(Base_Env):
    def __init__(self) -> None:
        super().__init__()
        # edit problem file
        self.curr_step= 0
        self.max_steps = 250
        self.pddl_env = None
        self.size = 5
        self.generator_max_steps = 9
        self.generator_action_dim = self.size* self.size
        self.domain = "../pddlgym/pddlgym/pddl/sokoban.pddl"
        self.problem_file = "../pddlgym/pddlgym/pddl/sokoban/problem00.pddl"
        # init parmas
        self.clear_env()
        self.step_order = ["choose_goal", "choose_player", "choose_stone"]  # else choose_walls
        self.pddl_env.reset()


    def init_pddl_env(self):
        create_env_file(self.problem_file, self.player_pos, self.stone_pos, self.goal_pos, self.walls)
        self.pddl_env, self.obs_shape, self.num_actions = get_pddl_env("sokoban")
       

    def get_max_episode_steps(self,):
        return 250


    def get_generator_max_steps(self,):
        return self.generator_max_steps


    def get_observation(self, agent=True):
        return self.pddl_env.get_obs().astype(np.float32) / 6


    def unpack_index(self, loc):
        col = int(loc % (self.size))
        row = int(loc / (self.size))
        return row+1, col+1


    def step_generator(self, loc):
        """The generator gets n_clutter + 2 moves to place the goal, agent, blocks.

        The action space is the number of possible squares in the grid. The squares
        are numbered from left to right, top to bottom.

        Args:
          loc: An integer specifying the location to place the next object which
            must be decoded into x, y playable_coordinates.

        Returns:
          Standard RL observation, reward (always 0), done, and info
        """
        step_order = self.step_order  # ["choose_goal", "choose_player" , "choose_stone"]  # else choose_walls
        current_turn = step_order[self.generator_step_count] if self.generator_step_count < len(step_order) else "place_walls"
        self.param_vec[self.generator_step_count] = loc
        if loc >= self.generator_action_dim:
            raise ValueError('Position passed to step_generator is outside the grid.')

        # Add offset of 1 for outside walls
        y, x = self.unpack_index(loc)

        done = False

        # Place goal
        if current_turn == "choose_goal":
            self.goal_pos = (x, y)

        # Place the agent
        elif current_turn == "choose_player":
            # Goal has already been placed here
            if self.goal_pos != (x, y):
                self.player_pos = (x, y)
            else:
                while True:
                    row = np.random.randint(self.size)+1
                    col = np.random.randint(self.size)+1
                    if self.goal_pos != (row, col):
                        self.player_pos = (row, col)
                        break

        elif current_turn == "choose_stone":
            if self.goal_pos != (x, y) and self.player_pos != (x, y):
                self.stone_pos = (x, y)
            else:
                while True:
                    row = np.random.randint(self.size)+1
                    col = np.random.randint(self.size)+1
                    if self.goal_pos != (row, col) and self.player_pos != (row, col):
                        self.stone_pos = (row, col)
                        break

        # Place wall
        elif self.generator_step_count < self.generator_max_steps:
            # If there is already an object there, action does nothing, also if it is on the grid bounderies
            if self.goal_pos != (x, y) and self.player_pos != (x, y) and self.stone_pos != (x, y):
                self.walls.append((x, y))

        self.generator_step_count += 1
        self.init_pddl_env()

        # End of episode

        if self.generator_step_count >= self.generator_max_steps:
            done = True
            self.reset()

        return self.get_observation(), 0, done, {}

    def get_observation_space(self):
        return self.obs_shape

    def get_generator_observation_space(self):
        return self.get_observation_space()

    def get_action_space(self):
        return gym.spaces.Discrete(self.num_actions)

    def get_action_dim(self):
        return self.num_actions

    def get_generator_action_space(self):
        return gym.spaces.Discrete(self.generator_action_dim) 

    def get_generator_action_dim(self):
        return self.generator_action_dim

    def reset(self,):
        self.curr_step= 0
        self.pddl_env.reset()
        return self.get_observation()

    def clear_env(self,):
        self.param_vec = np.zeros(self.generator_max_steps)
        self.generator_step_count = 0
        self.player_pos = (2, 2)
        self.goal_pos = (4, 4)
        self.stone_pos = (3, 3)
        self.walls = []
        self.init_pddl_env()
        return self.get_observation()


    def sample_random_state(self,):
        raise NotImplementedError

    def step(self, actions):
        action_types = [np.int, np.int32, np.int64]
        if type(actions) not in action_types:
            a = actions[0]
        else:
            a = actions
        s,r,d,_ =  self.pddl_env.step(a)
        self.curr_step+=1
        if self.curr_step> self.max_steps:
            d = True
            self.curr_step = 0
        return s,r,d,_

    def render(self,mode="rgb_array"):
        return (self.pddl_env.render("human_crisp")*255).astype(np.uint8)


register_env(Sokoban)


prefix = """(define (problem p024-microban-sequential) (:domain sokoban)
  (:objects
	dir-down - direction
	dir-left - direction
	dir-right - direction
	dir-up - direction
	player-01 - thing
	pos-1-1 - location
	pos-1-2 - location
	pos-1-3 - location
	pos-1-4 - location
	pos-1-5 - location
	pos-1-6 - location
	pos-2-1 - location
	pos-2-2 - location
	pos-2-3 - location
	pos-2-4 - location
	pos-2-5 - location
	pos-2-6 - location
	pos-3-1 - location
	pos-3-2 - location
	pos-3-3 - location
	pos-3-4 - location
	pos-3-5 - location
	pos-3-6 - location
	pos-4-1 - location
	pos-4-2 - location
	pos-4-3 - location
	pos-4-4 - location
	pos-4-5 - location
	pos-4-6 - location
	pos-5-1 - location
	pos-5-2 - location
	pos-5-3 - location
	pos-5-4 - location
	pos-5-5 - location
	pos-5-6 - location
	pos-6-1 - location
	pos-6-2 - location
	pos-6-3 - location
	pos-6-4 - location
	pos-6-5 - location
	pos-6-6 - location
	stone-01 - thing
  )
  (:goal (and
	(at-goal stone-01)))
  (:init\n"""

postfix = """	(is-player player-01)
	(is-stone stone-01)
	(move dir-down)
	(move dir-left)
	(move dir-right)
	(move dir-up)
"""
