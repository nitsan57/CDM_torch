
from Environments.environments import train_envs, get_all_avail_envs
from Models import fc, rnn
from Agents.dqn_agent import DQN_Agent
from Agents.ppo_agent import PPO_Agent
import utils
from Curriculum_managers.random_curriculum import Random_Curriculum
from Curriculum_managers.paired_curriculum import PAIRED_Curriculum
from Curriculum_managers.paired_curriculum_or import PAIRED_Curriculum_OR
from Curriculum_managers.paired_curriculum_hf import PAIRED_Curriculum_History_filter
from Curriculum_managers.paired_curriculum_hfe import PAIRED_Curriculum_History_filter_Entropy
from Curriculum_managers.paired_curriculum_or_hfe import PAIRED_Curriculum_Original_R_History_filter_Entropy
from Curriculum_managers.curriculum_no_regulator_he import Curriculum_Unregulated_Entropy_History
import argparse

def get_env(env_name):
    env = train_envs[env_name]()
    n_actions = env.action_space.n
    obs_shape = env.get_observation_space()

    return env, obs_shape, n_actions


parser = argparse.ArgumentParser()
parser.add_argument("--env_name", help=f"env_to_train - {get_all_avail_envs()}")
parser.add_argument("--method", help="the train method - random, paired, paired_entropy, entropy_only")
parser.add_argument("--iters", help="num_of_iters", type=int)


def main(args):
    print("starting... args=", args)
    env_names = get_all_avail_envs()
    env_names
    device = utils.init_torch()
    env_name = args.env_name
    env, obs_shape, n_actions = get_env(env_name)
    gen_obs_shape = env.get_generator_observation_space()
    gen_action_dim = env.get_generator_action_space().n


    p_agent = PPO_Agent(obs_shape, n_actions, device=device, batch_size=64, max_mem_size=10**5, lr=0.0001, model=rnn.RNN)
    teacher_agent = PPO_Agent(gen_obs_shape, gen_action_dim, device=device, batch_size=64, max_mem_size=10**5, lr=0.0001, model=rnn.RNN)
    
    if args.method == "random":
        teacher = Random_Curriculum(env ,trainee=p_agent)
    elif args.method == "paired":
        teacher = PAIRED_Curriculum(env, teacher_agent=teacher_agent ,trainee=p_agent)
    elif args.method == "paired_or":
        teacher = PAIRED_Curriculum_OR(env, teacher_agent=teacher_agent ,trainee=p_agent)
    elif args.method == "paired_hf":
        teacher = PAIRED_Curriculum_History_filter(env, teacher_agent=teacher_agent ,trainee=p_agent)
    elif args.method == "paired_hfe":
        teacher = PAIRED_Curriculum_History_filter_Entropy(env, teacher_agent=teacher_agent ,trainee=p_agent)
    elif args.method == "paired_or_hfe":
        teacher = PAIRED_Curriculum_Original_R_History_filter_Entropy(env, teacher_agent=teacher_agent ,trainee=p_agent)
    elif args.method == "no_regulator":
        teacher = Curriculum_Unregulated_Entropy_History(env, teacher_agent=teacher_agent ,trainee=p_agent)
        

    p_rewards = teacher.teach(n_iters=args.iters, n_episodes=1)


if __name__ == "__main__":

    args = parser.parse_args()
    main(args)




