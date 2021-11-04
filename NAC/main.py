#!/usr/bin/env python

"""
    This project, incorporates the episodic natural actor critic algorithm by
    Jan Peters and Stefan Schaal (https://doi.org/10.1016/j.neucom.2007.11.026)
    using Neural Networks and Tensorflow 1.9.

    This file is the key entry point. You can chose between different
    environments and specify their hyperparameters. These environments include
    all gym environments (https://gym.openai.com/) and all quanser environments
    (https://git.ias.informatik.tu-darmstadt.de/quanser/clients).

    Additionally you can chose if you want to use pretrained weights and if
    you want to train, evaluate and/or render and the environemnt.
"""

import sys
import tensorflow as tf
import numpy as np
import time
import gym
import os
import quanser_robots

import evaluation
import utilities

from my_environment import MyEnvironment
from critic import Critic
from actor import Actor
from nac import NAC

__author__ = "Maximilian A. Gehrke, Yannik P. Frisch, Tabea A. Wilke"
__data__ = "14.03.2019"
__copyright__ = "Copyright (C) 2019 Max Gehrke, Yannik Frisch, Tabea Wilke"
__credits__ = ["Maximilian A. Gehrke", "Yannik P. Frisch", "Tabea A. Wilke"]
__license__ = "GPL"
__version__ = "1.0"
__status__ = "Development"


# ---------------------- VARIABLES & CONSTANTS ------------------------------ #

# Load weights from file. Specify one of the following:
#   1. None, 2. 'best', 3. 'toplevel' (drag and drop the 'model'-folder to the
#   top level), 4. 'name_of_folder' (state the name, which is the saving time)
LOAD_WEIGHTS = None

# If you want to load weights which were trained on a different environment
# write the name in this variable (needed for real-robot testing).
LOAD_WEIGHTS_ENV = None

# What do we want to do?
TRAIN = True
EVALUATION = True
RENDER = True

# -------------------------- ENVIRONMENT ------------------------------------ #

# Select Environment
ENVIRONMENT = 1

"""
    0: Name of the Gym/Quanser environment.
    1: If the environment is descrete or continuous.
    2: Chose the discretization of continuous environments.
       If the environment is already discrete, put [0].
       you can also exactly specifiy the discretization.
    3: Batch size. How much steps should the agent perform before updating 
       parameters. If the trajectory ends before that (done == True), a new 
       trajectory is started.
    4: How many updates (of parameters) do we want.
    5: Discount factor for expected monte carlo return.
    6: Learning rate for actor model. sqrt(learning_rate/ Grad_j^T * F^-1).
    7: Learning rate for Adam optimizer in the critic model.
    8: The hidden layer size of the critic network.
"""

env_dict = {
    1: ['BallBalancerSim-v0', 'continuous', [3, 3],
        5000, 500, 0.99, 0.001, 0.01, 10],

    2:  ['Qube-v0', 'continuous', [5],
         5000, 500, 0.99, 0.001, 0.1, 10],

    3: ['CartpoleStabShort-v0', 'continuous',
        [[[-6.0], [-3.0], [0.0], [3.0], [6.0]]],
        2000, 300, 0.99, 0.001, 0.1, 10],

    4: ['CartpoleStabShort-v0', 'continuous', [3],
        2000, 300, 0.99, 0.001, 0.1, 10],

    5: ['CartpoleStabLong-v0', 'continuous', [3],
        500, 300, 0.97, 0.001, 0.1, 10],

    6: ['CartpoleStabRR-v0', 'continuous',
         [[[-6.0], [0.0], [6.0]]],
         500, 300, 0.99, 0.001, 0.1, 10],

    7: ['BallBalancerRR-v0', 'continuous', [3, 3],
        5000, 1000, 1, 0.001, 0.1, 10],

    8: ['CartPole-v0', 'discrete', [0],
        500, 300, 0.97, 0.001, 0.1, 10]
}

assert ENVIRONMENT in env_dict.keys()
env_details = env_dict[ENVIRONMENT]

# ------------------------ SESSION ------------------------------------------ #

sess = tf.InteractiveSession()

# ---------------------- GENERATE ENVIRONMENT ------------------------------- #

print("Generating {} environment:".format(env_details[0]))
env = MyEnvironment(env_details)

# ----------------------- GENERATE NETWORKS --------------------------------- #

start_time = time.time()
hour, mins, sec = time.strftime("%H,%M,%S").split(',')
print("Generating Neural Networks (Time: {}:{}:{}) ... "
      .format(hour, mins, sec))

if LOAD_WEIGHTS is not None:

    if LOAD_WEIGHTS == 'best':
        load_env = env.name
        if LOAD_WEIGHTS_ENV is not None:
            load_env = LOAD_WEIGHTS_ENV
        model_dir, best_reward, best_date = utilities.find_best_model(load_env)
        print("\tLoading best model from {} with a reward of {}!"
              .format(best_date, best_reward))
    elif LOAD_WEIGHTS == "toplevel":
        model_dir = 'model/'  # toplevel
        print("\tLoading model from toplevel!")
    else:
        model_dir = "data/" + env.name + "/" + str(LOAD_WEIGHTS) + "/model/"
        print("\tLoading model from {}!".format(LOAD_WEIGHTS))
    print("\t", end='')
    sys.stdout.flush()

    saver = tf.train.import_meta_graph(model_dir + 'nac_model.meta')
    saver.restore(sess, tf.train.latest_checkpoint(model_dir))
    graph = tf.get_default_graph()

    a_state_input = graph.get_tensor_by_name("actor/state_input:0")
    a_actions_input = graph.get_tensor_by_name("actor/actions_input:0")
    a_advantages_input = \
        graph.get_tensor_by_name("actor/advantages_input:0")
    a_probabilities = graph.get_tensor_by_name("actor/probabilities:0")
    a_weights = graph.get_tensor_by_name("actor/weights:0")

    c_state_input = graph.get_tensor_by_name("critic/state_input:0")
    c_true_vf_input = graph.get_tensor_by_name("critic/true_vf_input:0")
    c_output = graph.get_tensor_by_name("critic/output:0")
    c_optimizer = tf.get_collection("optimizer")
    c_loss = graph.get_tensor_by_name("critic/loss:0")

else:
    a_state_input, a_actions_input, a_advantages_input, \
        a_probabilities, a_weights = Actor.create_policy_net(env)

    c_state_input, c_true_vf_input, c_output, c_optimizer, c_loss = \
        Critic.create_value_net(env)

actor = Actor(env, a_state_input, a_actions_input, a_advantages_input,
              a_probabilities, a_weights)
critic = Critic(env, c_state_input, c_true_vf_input, c_output,
                c_optimizer, c_loss)
nac = NAC(env, actor, critic)

if LOAD_WEIGHTS is None:
    sess.run(tf.global_variables_initializer())

env.network_generation_time = int(time.time() - start_time)
print("Done! (Time: " + str(env.network_generation_time) + " seconds)")

# ----------------------- TRAINING NETWORKS --------------------------------- #

if TRAIN:
    start_time_training = time.time()

    max_rewards = []
    cum_batch_traj_rewards = []
    mean_batch_traj_rewards = []
    total_episodes = []
    times = []

    for u in range(env.num_of_updates):
        start_time = time.time()

        # Act in the env and update weights after collecting data
        batch_traj_rewards = nac.run_batch(sess)

        print('Update {} with {} trajectories with rewards of: {}'
              .format(u, len(batch_traj_rewards), batch_traj_rewards))

        max_rewards.append(np.max(batch_traj_rewards))
        cum_batch_traj_rewards.append(np.sum(batch_traj_rewards))
        mean_batch_traj_rewards.append(np.mean(batch_traj_rewards))
        total_episodes.append(len(batch_traj_rewards))
        times.append(time.time() - start_time)

    env.network_training_time = int(time.time() - start_time_training)

    try:
        os.makedirs(env.save_folder)
        os.makedirs(env.save_folder + "/training/")
    except FileExistsError:
        pass

    # Save training data
    np.save('{}/training/max_update_reward'.format(env.save_folder),
            max_rewards)
    np.save('{}/training/cum_update_reward'.format(env.save_folder),
            cum_batch_traj_rewards)
    np.save('{}/training/mean_traj_update_reward'.format(env.save_folder),
            cum_batch_traj_rewards)
    np.save('{}/training/total_episodes'.format(env.save_folder),
            total_episodes)

    # Plot training rewards
    evaluation.plot_training_rewards(
        env, cum_batch_traj_rewards, mean_batch_traj_rewards)

    # Save model to file
    saver = tf.train.Saver()
    saver.save(sess, '{}/model/nac_model'.format(env.save_folder))

# ------------------- EVALUATE | RENDER ------------------------------ #

if EVALUATION:
    evaluation.evaluate(env, sess, actor)

if RENDER:
    evaluation.render(env, sess, actor)

sess.close()
