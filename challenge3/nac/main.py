import tensorflow as tf
import numpy as np
import sys
import time
import gym
import quanser_robots

from nac.my_environment import MyEnvironment
from nac.critic import value_gradient
from nac.actor import Actor
from nac.nac import run_batch
from nac import evaluation

# ----------------------------- GOALS ------------------------------------- #

# Environments which have to be solved:
# DoubleCartPole: "DoublePendulum-v0"
# FurutaPend: "Qube-v0"
# BallBalancer: "BallBalancerSim-v0"
# Levitation: "Levitation-v1"

# ----------------------------- TODOS --------------------------------------- #
# TODO: Continuous actions
# TODO: Improvements: https://github.com/rgilman33/simple-A2C/blob/master/3_A2C-nstep-TUTORIAL.ipynb
# TODO: Some paper said, not taking batches, where we only had 1 episode because
#   it does not yield any improvements to update the weights of perffect episodes.

# ---------------------- VARIABLES & CONSTANTS ------------------------------ #
# Select Rendering
RENDER = True

# Select how we treat actions
# IMPORTANT: Only False works yet
CONTINUOUS = False
HIDDEN_LAYER_SIZE = 10

# Select complexity of policy network
# IMPORTANT: Only False works yet
COMPLEX_POLICY_NET = False

# Load weights from file and use them
LOAD_WEIGHTS = True # TODO

# Select Environment
ENVIRONMENT = 4

"""
    0: Name of the Gym/Quanser environment.
    1: If the environment is descrete or continuous.
    2: Chose the discretization of continuous environments (discrete = 0).
       Only important, if CONTINUOUS = False.
    3: Batch size. How much steps should the agent perform before updating 
       parameters. If the trajectory ends before that (done == True), a new 
       trajectory is started.
    4: How many updates (of parameters) do we want.
    5: Discount factor for expected monte carlo return.
    6: Learning rate for actor model. sqrt(learning_rate/ Grad_j^T * F^-1).
    7: Learning rate for Adam optimizer in the critic model.
"""

env_dict = {1: ['CartPole-v0',          'discrete',     [0],    500, 300, 0.97, 0.001, 0.1],

            2: ['DoublePendulum-v0',    'continuous',   [3],    200, 300, 0.97, 0.001, 0.1],
                # Does not diverge with batch size of 2000

            3: ['Qube-v0',              'continuous',   [3],    200, 300, 0.97, 0.001, 0.1],
            4: ['BallBalancerSim-v0',   'continuous',   [5, 5], 2000, 300, 0.97, 0.001, 0.1],
            5: ['Levitation-v1',        'continuous',   [3],    200, 300, 0.97, 0.001, 0.1],
            6: ['Pendulum-v0',          'continuous',   [3],    200, 300, 0.97, 0.001, 0.1],
            7: ['CartpoleStabRR-v0',    'continuous',   [3],    200, 300, 0.97, 0.001, 0.1]}

assert ENVIRONMENT in env_dict.keys()
env_details = env_dict[ENVIRONMENT]


# ---------------------- GENERATE ENVIRONMENT ------------------------------- #
print("Generating {} environment:".format(env_details[0]))
env = MyEnvironment(env_details, CONTINUOUS,
                    COMPLEX_POLICY_NET, HIDDEN_LAYER_SIZE)

# Initialize the session
sess = tf.InteractiveSession()

# ----------------------- GENERATE NETWORKS --------------------------------- #

print("Generating Neural Networks ... ", end="")
start_time = time.time()
sys.stdout.flush()
actor = Actor(env)
value_grad = value_gradient(env)
env.network_generation_time = int(time.time() - start_time)
print("Done! (Time: " + str(env.network_generation_time) + " seconds)")

sess.run(tf.global_variables_initializer())

# ----------------------- TRAINING NETWORKS --------------------------------- #

max_rewards = []
total_episodes = []
times = []

for u in range(env.num_of_updates):
    start_time = time.time()

    # Act in the env and update weights after collecting data
    reward, n_episodes = \
        run_batch(env, actor, value_grad, sess, u)

    max_rewards.append(np.max(reward))
    total_episodes.append(n_episodes)
    times.append(time.time() - start_time)
print('Average time: %.3f' % (np.sum(times) / env.num_of_updates))

# ------------------- EVALUATE | SAVE | RENDER ------------------------------ #

evaluation.evaluate(env, sess, actor)

saver = tf.train.Saver()
saver.save(sess, '{}/model/nac_model'.format(env.save_folder))

if RENDER:
    evaluation.render(env, sess, actor)

sess.close()
