
from Environments.environments import all_envs, get_all_avail_envs
from Models import fc, rnn
from Agents.dqn_agent import DQN_Agent
from Agents.ppo_agent import PPO_Agent
import utils
from Curriculum_managers.random_curriculum import Random_Curriculum
from Curriculum_managers.paired_curriculum import PAIRED_Curriculum
# from Curriculum_managers.paired_curriculum_extented import  PAIRED_Curriculum_entropy
from Curriculum_managers.paired_curriculum_no_regret import  PAIRED_Curriculum_no_regret
from Curriculum_managers.paired_curriculum_no_regret_teacher_entropy import  PAIRED_Curriculum_no_regret_entropy
import argparse

def get_env(env_name):
    env = all_envs[env_name](random_reset_loc=True)
    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape
    return env, obs_shape, n_actions


parser = argparse.ArgumentParser()
parser.add_argument("--env_name", help=f"env_to_train - {get_all_avail_envs()}")
parser.add_argument("--method", help="the train method - random, paired, paired_entropy, paired_no_reg")
parser.add_argument("--iters", help="num_of_iters", type=int)


def main(args):
    print("starting... args=", args)
    env_names = get_all_avail_envs()
    env_names
    device = utils.init_torch()
    env_name = args.env_name
    env, obs_shape, n_actions = get_env(env_name)
    gen_obs_shape = env.get_generator_observation_space().shape
    gen_action_dim = env.get_generator_action_space().n

    p_agent = DQN_Agent(obs_shape, n_actions, device=device, batch_size=64, max_mem_size=10**5, exploration_epsilon=0.3, eps_dec=0, lr=0.001, model=rnn.RNN)
    teacher_agent = DQN_Agent(gen_obs_shape, gen_action_dim, device=device, batch_size=64, max_mem_size=10**5, exploration_epsilon=1, eps_dec=0.0001, lr=0.001, model=rnn.RNN)
    

    if args.method == "random":
        teacher = Random_Curriculum(env ,trainee=p_agent)
    elif args.method == "paired":
        teacher = PAIRED_Curriculum(env, teacher_agent=teacher_agent ,trainee=p_agent)
    elif args.method == "paired_entropy":
        teacher = PAIRED_Curriculum_no_regret_entropy(env, teacher_agent=teacher_agent ,trainee=p_agent)
    elif args.method == "paired_no_reg":
        teacher = PAIRED_Curriculum_no_regret(env, teacher_agent=teacher_agent ,trainee=p_agent)

    p_rewards = teacher.teach(n_iters=args.iters, n_episodes=8)


if __name__ == "__main__":

    args = parser.parse_args()
    main(args)




