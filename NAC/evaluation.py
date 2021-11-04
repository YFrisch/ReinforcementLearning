#!/usr/bin/env python

import time
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

__author__ = "Maximilian A. Gehrke, Yannik P. Frisch, Tabea A. Wilke"
__data__ = "14.03.2019"
__copyright__ = "Copyright (C) 2019 Max Gehrke, Yannik Frisch, Tabea Wilke"
__credits__ = ["Maximilian A. Gehrke", "Yannik P. Frisch", "Tabea A. Wilke"]
__license__ = "GPL"
__version__ = "1.0"
__status__ = "Development"


def evaluate(env, sess, actor, episodes=100):
    """
    Evaluates the learned agent on a given instance of MyEnvironment.
    Evaluation is done by calculating cumulative rewards of each trajectory and
    the average reward in each trajectory. These are saved as text
    (results.txt) and as plots under "/data/" in a folder named with a current
    time stamp. Additionally all hyperparameters are saved to
    "hyperparameters.txt".

    :param env: MyEnvironment instance; The environment for agent evaluation
    :param sess: Tensorflow session
    :param actor: The actor (policy network) used for generating actions
    :param episodes: Number of episodes used for evaluation
    :return: None
    """

    # Create folder for saving
    try:
        os.makedirs(env.save_folder)
    except FileExistsError:
        pass

    # -------------------- EVALUATION OF ENVIRONMENT ------------------------ #

    print("\nEVALUATION: {} episodes on {} until 'done'!"
          .format(episodes, env.name))

    cumulative_episode_reward = []
    average_episode_reward = []
    trajectory_lengths = []
    list_of_all_rewards = []

    for e in range(episodes):

        done = False
        observation = env.reset()

        undiscounted_return = 0
        rewards = []
        t = 0
        while True:

            if done:
                trajectory_lengths.append(t)
                cumulative_episode_reward.append(undiscounted_return)
                average_episode_reward.append(np.mean(rewards))
                t = 0
                break

            action, _ = actor.get_action(sess, observation)
            observation, reward, done, _ = env.step(action)

            undiscounted_return += reward
            rewards.append(reward)
            t += 1

        list_of_all_rewards.append(rewards)

    print("Average trajectory length:", np.mean(trajectory_lengths))
    print("Average episode reward:", np.mean(average_episode_reward))
    print("Average cumulative episode reward:",
          np.mean(cumulative_episode_reward))

    # ------------------------- SAVE RESULTS -------------------------------- #

    results_file = open("{}/results.txt".format(env.save_folder), 'w')

    results_string = \
        "Average trajectory length: {}\n" \
        "Average transition reward: {}\n" \
        "Average trajectory reward: {}\n" \
            .format(np.mean(trajectory_lengths),
                    np.mean(cumulative_episode_reward),
                    np.mean(average_episode_reward))

    results_file.write(results_string)
    results_file.close()

    # Plot results
    plot_save_rewards(env, list_of_all_rewards,
                      cumulative_episode_reward, average_episode_reward)

    # -------------------- SAVE HYPERPARAMETERS ----------------------------- #

    param_file = open("{}/parameters.txt".format(env.save_folder), 'w')

    param_string = \
        "Environment name: {}\n"\
        "Is Env discrete/continuous: {}\n"\
        "Do we use continuous actions (or discretize): {}\n"\
        "Learning Epochs/num of Updates: {}\n" \
        "Batch size: {}\n" \
        "Discount factor for monte carlo return: {}\n"\
        "Network generation time: {} seconds\n"\
        "Network training time: {} seconds\n"\
        "Learning rate actor: {}\n"\
        "Learning rate for Adam optimizer in critic: {}\n"\
        "Critic hidden layer size: {}"\
        .format(env.name, env.action_form, env.discretization,
                env.num_of_updates, env.time_steps, env.mc_discount_factor,
                env.network_generation_time, env.network_training_time,
                env.learning_rate_actor, env.learning_rate_critic,
                env.hidden_layer_critic)
    param_file.write(param_string)
    param_file.close()


def plot_save_rewards(env, rewards, cumulative_episode_reward,
                      average_episode_reward):
    """
    Plot and save results of evaluation.

    :param rewards: an array of one array for each episode with the reward per
        timestep
    :param env: Instance of MyEnvironment class; used to name the folder for
        saving plots
    :param cumulative_episode_reward: 2D-array, containing all episodes and
        the cumulative reward per step per episode
    :param average_episode_reward: 2D-array, episode x average reward per step
        per episode
    :return: None
    """

    mean_avg_rew = np.mean(average_episode_reward)
    mean_cum_rew = np.mean(cumulative_episode_reward)
    episodes = len(average_episode_reward)

    # Plot & save average reward per episode
    plt.figure()
    plt.title("Average reward per episode")
    plt.xlabel("Episode")
    plt.ylabel("Average reward")
    plt.plot(average_episode_reward, label="Mean per episode")

    y_mean = [mean_avg_rew] * episodes
    plt.plot(range(episodes), y_mean,
             label='Overall mean (' + str(mean_avg_rew) + ')', linestyle='--')

    plt.legend(loc='upper right')
    plt.savefig("{}/avg_reward.png".format(env.save_folder))
    plt.close()

    # Plot & save cumulative reward per episode
    plt.figure()
    plt.title("Cumulative reward per episode")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.plot(cumulative_episode_reward, label="Mean per episode")

    y_mean = [mean_cum_rew] * episodes
    plt.plot(range(episodes), y_mean,
             label='Overall mean (' + str(mean_cum_rew) + ')', linestyle='--')

    plt.legend(loc='upper right')
    plt.savefig("{}/cum_reward.png".format(env.save_folder))
    plt.close()

    # Mean reward per timestep
    dat = pd.DataFrame()
    rew = pd.DataFrame(rewards).transpose()

    for x in range(0, np.shape(rew)[1]):
        # Creating a dataframe with the relevant columns
        dat2 = pd.DataFrame()
        dat2['reward'] = rew[x]
        dat2['timesteps'] = np.linspace(0, np.shape(rew[x])[0]-1,
                                        np.shape(rew[x])[0], dtype=int)
        dat2['Episode'] = np.repeat(x, np.shape(rew[x]))
        dat = dat.append(dat2, ignore_index=True)

    # Plotting the mean with standard deviation
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.lineplot(x='timesteps', y='reward', data=dat, ax=ax, estimator='mean',
                 ci='sd')
    plt.show()
    ax.set_title('Rewards per time step')
    fig.savefig("{}/time_step_reward.png".format(env.save_folder),
                bbox_inches='tight')


def render(env, sess, actor, episodes=10):
    """
    Renders the learned agent on a given instance of MyEnvironment class.

    :param env: MyEnvironment instance, on which the agent is evaluated on
    :param sess: Tensorflow session
    :param actor: The actor (policy network) used for generating actions
    :param episodes: Number of episodes used for rendering
    :return: None
    """

    time_steps = 10000

    print("\nRENDER: {} episodes on {} until 'done'!"
          .format(episodes, env.name))

    for e in range(episodes):

        print("Episode {} ... ".format(e), end='')
        sys.stdout.flush()

        done = False
        observation = env.reset()

        for t in range(time_steps):

            # Render environment
            env.render()
            time.sleep(0.01)

            # Break loop, if episode has finished
            if done:
                print("Episode ended after {} time steps!".format(t))
                break

            action, _ = actor.get_action(sess, observation)
            action = action
            observation, _, done, _ = env.step(action)



def plot_training_rewards(env, cum_batch_traj_rewards,
                          mean_batch_traj_rewards):
    """
    Plot the rewards of the training of the environment

    :param env: MyInvironment instance, needed for naming the files
    :param cum_batch_traj_rewards: an array where each number stands for one
        batch and is defined by the sum of the cumulative trajectory rewards
    :param mean_batch_traj_rewards: an array where each number stands for one
        batch and is defined by the mean of the cumulative trajectory rewards
    :return: None
    """

    # Plot/Save average reward per episode
    plt.figure()
    plt.title("Average trajectory reward per update")
    plt.xlabel("Update")
    plt.ylabel("Average reward")
    plt.plot(mean_batch_traj_rewards)
    plt.savefig('{}/training/mean_traj_update_reward.png'
                .format(env.save_folder))
    plt.close()

    # Plot/Save cumulative reward per episode
    plt.figure()
    plt.title("Cumulative reward per update")
    plt.xlabel("Update")
    plt.ylabel("Cumulative reward")
    plt.plot(cum_batch_traj_rewards)
    plt.savefig('{}/training/cum_update_reward.png'
                .format(env.save_folder))
    plt.close()
