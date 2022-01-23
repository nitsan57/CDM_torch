
import pickle
from Environments.environments import get_all_avail_test_envs, test_envs
from Environments.adversarial import AdversarialEnv
from Models import fc, rnn
from Agents.dqn_agent import DQN_Agent
from Agents.ppo_agent import PPO_Agent
import utils
from Curriculum_managers.paired_curriculum import PAIRED_Curriculum
import argparse
from matplotlib import pyplot as plt

NUM_ITERS = 1


# Lint as: python3
"""Tests trained models on transfer environments to generate videos and scores.

Note that this code assumes it will be provided with a .csv file indicating
which checkpoints it should load based on finding the best hyperparameters
for a given metric, such as 'SolvedPathLength_last20%'. It assumes this csv will
have columns labeled 'metric', 'exp_id', 'best_seeds', and 'settings'. Such a
csv can be created using the function utils.save_best_work_units_csv()
"""
import os


from absl import app
import numpy as np
import pandas as pd
pd.options.plotting.backend = "plotly"
import tensorflow as tf  # tf
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PATH = "./results"

def get_test_env(env_name):
    env = test_envs[env_name]()
    n_actions = env.action_space.n
    obs_shape = env.observation_space
    return env, obs_shape, n_actions


def run_agent(agent, env, num_iters=1):
    agent.set_eval_mode()
    mean_r = 0
    for i in range(num_iters):
        obs = env.reset()
        # break
        R = 0
        t = 0
        while True:
            action = agent.act(obs)
            obs, r, done, _ = env.step(action)

            R += r
            t += 1
            reset = t == 500

            if done or reset:
                break
        mean_r +=R
    return mean_r / num_iters


def display_vid(frames, title):
    img = None
    plt.clf()

    plt.title(title)
    for f in frames:
        if img is None:
            img = plt.imshow(f)
        else:
            img.set_data(f)
        
        plt.pause(.1)
        plt.draw()



def run_policy(i, env_name, policy):
    env, obs_shape, n_actions = get_test_env(env_name)
    for j in range(i):
        env.reset()

    return run_agent(policy, env)


def run_experiments_on_env(agent_list, env_name,all_rewards, num__iters=NUM_ITERS, display_vid=False):
    exp_names = []
    all_vids = {agent_name:[] for agent_name in agent_list}
    for i in range(num__iters):
        if i in BLACK_LIST_ENV_IDX[env_name]:
            # IF UN POSSIBLE ENV SKIP
            continue
        exp_name = env_name + "_" + str(i)
        exp_names.append(exp_name)


        for agent_name in agent_list:
            reward, images = run_policy(i, env_name, agent_list[agent_name])
            reward = np.round(reward, 3)
            all_rewards[agent_name].append(reward)
            all_vids[agent_name].append(images)

        # remove 0 for-all reward results
        num_empty_rewards = 0
        for a_n , r_l in all_rewards.items():
            if r_l[-1] == 0:
                num_empty_rewards +=1

        if num_empty_rewards == len(all_rewards.keys()):
            exp_names = exp_names[:-1]
            for agent_name in agent_list:
                all_rewards[agent_name] = all_rewards[agent_name][:-1]
                all_vids[agent_name] = all_vids[agent_name][:-1]

        else:
            print('------------------')
            for agent_name in agent_list:

                if display_vid:
                    if all_rewards[agent_name] != 0:
                        display_vid(all_vids[agent_name], agent_name)

                print(f"{agent_name} reward: {all_rewards[agent_name][-1]} on env: {exp_name}")

    return exp_names, all_rewards

def compare_all_envs_train_rewards(all_env_names, show=False):
    env_idx = 1

    agent_names = ['original', 'entropy', 'history']
    agent_list = {}
    for a_n in agent_names:
        path = PATH + f"{a_n}_policy_latest"
        policy = tf.compat.v2.saved_model.load(path)
        agent_list[a_n] = policy

    all_names= []
    all_rewards = {agent_name:[] for agent_name in agent_list}

    for env_name in all_env_names:
        exp_names, all_rewards = run_experiments_on_env(agent_list, env_name, all_rewards)
        all_names += exp_names
    
    pd_dict = {a_n: all_rewards[a_n] for a_n in all_rewards}
    pd_dict['env_names'] = all_names
    y = [a_n for a_n in all_rewards]
    
    df = pd.DataFrame(pd_dict)
    fig = px.line(df, x="env_names", y=y, title='Reward compare')
    if show:
        fig.show()

def run_experiments_on_env_avg_reward(agent_list, env_name, all_rewards, num__iters=NUM_ITERS, display_vid=False):
    all_vids = {agent_name:[] for agent_name in agent_list}
    num_envs_trained = 0
    for i in range(num__iters):
        if i in BLACK_LIST_ENV_IDX[env_name]:
            # IF UN POSSIBLE ENV SKIP
            continue
        num_envs_trained +=1

        exp_name = env_name + "_" + str(i)


        for agent_name in agent_list:
            reward, images = run_policy(i, env_name, agent_list[agent_name])
            all_rewards[env_name][agent_name] +=reward
            all_vids[agent_name].append(images)
    
    for agent_name in agent_list:
        all_rewards[env_name][agent_name] = np.round(all_rewards[env_name][agent_name] /num_envs_trained, 3)

    return all_rewards



def compare_final_weights_avg_reward(all_env_names, agent_names = ['original', 'entropy', 'history'], show=False):

    agent_list = {}
    for a_n in agent_names:
        path = PATH + f"{a_n}_policy_latest"
        policy = tf.compat.v2.saved_model.load(path)
        agent_list[a_n] = policy

    all_names= []
    all_rewards = {env_name:{} for env_name in all_env_names}
    for agent_name in agent_list:
        for env_name in all_env_names:
            all_rewards[env_name][agent_name] = 0

    for env_name in all_env_names:
        all_rewards = run_experiments_on_env_avg_reward(agent_list, env_name, all_rewards)
        all_names.append(env_name)
    
    titles = [env.replace("MultiGrid-", "").replace("Minigrid-", "") for env in ALL_ENVS]

    color10_16 = ['blue', 'cyan', 'red', "yellow",  "green",  "orange"]
    fig = make_subplots(rows=(len(all_env_names)//4+1), cols=len(all_env_names), subplot_titles=titles)
    for i,env_name in enumerate(all_env_names):
        fig.add_trace(
            go.Bar(x=[agent_name for agent_name in agent_list], y=[all_rewards[env_name][agent_name] for agent_name in agent_list],name=env_name,marker_color=color10_16),row=((i//4)+1), col=((i%4)+1))


    fig.update_layout(height=600, width=1600, title_text="Reward Comparation", showlegend=False, legend=dict(
    yanchor="bottom",
    y=-0.5,
    xanchor="right",
    x=1
))
    if show:
        fig.show()
    fig.write_html('final_reward.html')



def run_experiments_on_env_curriculum_avg_reward(agent, env_name, num__iters=NUM_ITERS, display_vid=False):
    num_envs_trained = 0
    avg_reward = 0
    for i in range(num__iters):
        if i in BLACK_LIST_ENV_IDX[env_name]:
            # IF UN POSSIBLE ENV SKIP
            continue
        num_envs_trained +=1

        reward, images = run_policy(i, env_name, agent)
        avg_reward +=reward
    
        avg_reward = np.round(avg_reward /num_envs_trained, 3)

    return avg_reward



def compare_curriculum_weights_avg_reward(all_env_names, agent_names, agent_model, trained_env_name, avg_only=False, show=False):
    path = PATH 
    
    all_rewards = {a_n:{e_n: [] for e_n in all_env_names} for a_n in agent_names}
    x_axis = {a_n:{e_n: [] for e_n in all_env_names} for a_n in agent_names}
    agent  = agent_model
    

    # save_format = f"_{}_antagonist.ckpt"


    for a_n in agent_names:
        parent_folder = os.path.join(path, a_n, trained_env_name)
        agent_weight_dirs = [ os.path.join(parent_folder, name) for name in os.listdir(parent_folder) if ("trainee" in name) ]
        agent_weight_dirs = sorted(agent_weight_dirs, key=lambda x: int(x.split("_")[-2]))

        for env_name in all_env_names:
            for a_w in agent_weight_dirs:
                agent.load_agent(a_w)
                policy_iter = a_w.split("_")[-2]
                x_axis[a_n][env_name].append(policy_iter)
                full_name = a_n + "_" + policy_iter
                avg_reward = run_experiments_on_env_curriculum_avg_reward(agent, env_name)
                all_rewards[a_n][env_name].append(avg_reward)
                logger(f"finished agent {full_name} on {env_name} reward: {avg_reward}")
    


    if avg_only:
        fig = make_subplots(rows=len(agent_names), cols=1, subplot_titles=agent_names)
        for i,a_n in enumerate(agent_names):
            envs_dict = all_rewards[a_n]
            envs_curriculum_vals = pd.DataFrame(envs_dict).values
            avg_reward_graph = np.average(envs_curriculum_vals, axis=1)

    
            fig.add_trace(
                go.Line(x=x_axis[a_n][all_env_names[-1]], y=avg_reward_graph ,name=a_n),row=(i+1), col=(1))


        fig.update_layout(height=600, width=1600, title_text="Reward Comparation")
        
        fig.write_html('avg_curriculum_reward.html')

    else:
        color10_16 = ['blue', 'cyan', 'red', "yellow",  "green",  "orange"]
        fig = make_subplots(rows=len(agent_names), cols=len(all_env_names), subplot_titles=all_env_names)
    
        for i,env_name in enumerate(all_env_names):
            for j,agent_name in enumerate(agent_names):
            # print("!!!!", len(x_axis[agent_name])), len(all_rewards[agent_name])
                fig.add_trace(
                    go.Line(x=x_axis[agent_name][env_name], y=all_rewards[agent_name][env_name],name=agent_name,marker_color=color10_16),row=(j+1), col=(i+1))


        fig.update_layout(height=600, width=1600, title_text="Reward Comparation")
        fig.write_html('all_env_curriculum_reward.html')

    if show:
        fig.show()
    return all_rewards


def plot_results(domain_name, agent_names, weights_num, difficulties, results):    

    for difficulty in difficulties:        
        fig = pd.DataFrame(data=results[difficulty], index=agent_names, columns=weights_num).T.plot()
        fig.write_html(f"{domain_name}_{difficulty}.html")

        smoothed = np.apply_along_axis(lambda m: np.convolve(m, np.ones(3)/3, mode='valid'), axis=1, arr=results[difficulty])
        fig = pd.DataFrame(data=smoothed[:len(smoothed)], index=agent_names, columns=weights_num[:len(smoothed[0])]).T.plot()
        fig.write_html(f"{domain_name}_{difficulty}_smooth.html")



def load_spaces(domain_name):
    if domain_name == "Maze":
        env = AdversarialEnv()
    else:
        env_names = get_all_avail_test_envs(domain_name, "easy")
        env = test_envs[domain_name]["easy"][env_names[0]]
    n_actions = env.get_action_space().n
    obs_shape = env.get_observation_space()
    return obs_shape, n_actions
    

def cacl(domain_folder_name, domain_name, agent_names, weights_num, difficulties, num_iters):
    device = utils.init_torch()
    
    obs_shape, n_actions = load_spaces(domain_name)
    agent = PPO_Agent(obs_shape, n_actions, device=device, batch_size=64, max_mem_size=10**5, lr=0.0001, model=rnn.RNN)

    all_results = {k:[list()]*len(agent_names) for k in difficulties}
    for i,agent_name in enumerate(agent_names):
        for j,w in enumerate(weights_num):
            f_name = f"./results/{agent_name}/{domain_folder_name}/_{w}_trainee.ckpt"
            agent.load_agent(f_name)
            for difficulty in difficulties:
                env_names = get_all_avail_test_envs(domain_name, difficulty)
                # mean_reward = 0
                for env_name in env_names:
                    env = test_envs[domain_name][difficulty][env_name]() #test_envs[domain_name][difficulty][env_names]()
                    reward = run_agent(agent, env,num_iters=num_iters)
                    # mean_reward +=reward
                    all_results[difficulty][i].append(reward)
                # mean_reward /= len(env_names)
                # print(f"{difficulty}: {agent_name} :{mean_reward}")
                # all_results[difficulty][i][j] = mean_reward
    return all_results



def main(args):
    difficulties = ["easy", "medium", "hard"]
    agent_names = ['Random_Curriculum','PAIRED_Curriculum', 'PAIRED_Curriculum_OR', 'PAIRED_Curriculum_History_filter', 'PAIRED_Curriculum_History_filter_Entropy', 'PAIRED_Curriculum_Original_R_History_filter_Entropy', 'Curriculum_Unregulated_Entropy_History']
    domain_name = "Maze"
    domain_folder_name = "MiniAdversarialEnv"


    last_ckpt = 139999
    ckpt_diff = 5000
    num_iters = 2
    weights_num = list(range(last_ckpt,ckpt_diff,-ckpt_diff))[::-1]

    all_results = cacl(domain_folder_name, domain_name, agent_names, weights_num, difficulties, num_iters)
    with open('results.pickle', 'wb') as handle:
        pickle.dump(all_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # plot_results(domain_name, agent_names, weights_num, difficulties, all_results)


if __name__ == '__main__':
    main(None)
