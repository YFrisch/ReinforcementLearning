#!/usr/bin/env python

import tensorflow as tf
import gym
import quanser_robots
import numpy as np
import random

__author__ = "Maximilian A. Gehrke, Yannik P. Frisch, Tabea A. Wilke"
__data__ = "14.03.2019"
__copyright__ = "Copyright (C) 2019 Max Gehrke, Yannik Frisch, Tabea Wilke"
__credits__ = ["Maximilian A. Gehrke", "Yannik P. Frisch", "Tabea A. Wilke"]
__license__ = "GPL"
__version__ = "1.0"
__status__ = "Development"


class Actor:
    def __init__(self, env, state_input, actions_input, advantages_input,
                 probabilities, trainable_vars):
        """
        Construct the critic of our natural actor critic algorithm.

        :param env: a MyEnvironment instance
        :param state_input: a tensorflow placeholder which we will feed the
            states of our batch
        :param actions_input: a tensorflow placeholder which we will feed the
            actions of our batch which we took in the corresponding state
        :param advantages_input: a tensorflow placeholder which we will feed
            the advantages of our states of the batch
        :param probabilities: a tensorflow variable which holds the output of
            our actor network. The output is an array for each action in our
            actions space which contains a value between 0 and 1 that sums up
            to one and represents the probability that the action should be
            taken.
        :param trainable_vars: a tensorflow variable which contains all the
            variables of our actor network which can be trained

        """

        self.env = env
        self.state_input = state_input
        self.actions_input = actions_input
        self.advantages_input = advantages_input
        self.probabilities = probabilities
        self.trainable_vars = trainable_vars

    def update(self, sess, batch_states, batch_actions, batch_advantages):
        """
        Update the weights of our critic.

        :param sess: a tensorflow session
        :param batch_states: an array containing all the states of a batch
        :param batch_actions: an array containing all the actions of a batch
            which we took at the corresponding states
        :param batch_advantages: an array containing all the advantages of
            all the states of a batch
        :return: None
        """
        sess.run(self.trainable_vars,
                 feed_dict={self.state_input: batch_states,
                            self.actions_input: batch_actions,
                            self.advantages_input: batch_advantages})

    def get_action(self, sess, observation):
        """
        Get an action using the current state of our actor network. We input
        the given observation and get a probability for each possible action
        that we can take. Then we sample one of the actions (w.r.t the
        probabilites) and return it.

        :param sess: a tensorflow session
        :param observation: an observation of our environment
        :return: an action; it's index in env.action_space
        """

        # Get probabilites of actions from our actor given the observation
        obs_vector = np.expand_dims(observation, axis=0)
        probs = sess.run(
            self.probabilities,
            feed_dict={self.state_input: obs_vector})

        # Stochastically generate an action using the probabilites of the actor
        probs_sum = 0
        action_i = None  # Action index
        rnd = random.uniform(0, 1)
        for k in range(len(self.env.action_space)):
            probs_sum += probs[0][k]
            if rnd < probs_sum:
                action_i = k
                break
            elif k == (len(self.env.action_space) - 1):
                action_i = k
                break

        return self.env.action_space[action_i], action_i

    def get_net_variables(self):
        """
        This method returns the variables of our policy network. It shall only
        be used for development purposes. If development has finished,
        a new method imbedding the desired functionality shall be created.

        :return: placeholder state variable,
            placeholder action variable,
            placeholder advantages variable,
            network output variable (= probabilites of actions),
            trainable weights variable
        """

        return self.state_input, self.actions_input, \
               self.advantages_input, self.probabilities, \
               self.trainable_vars

    @staticmethod
    def create_policy_net(env):
        """
        Neural Network to approximate our policy.

        Estimating: We want to know the probabilities the network outputs for a
        state. Just feed the state you want to know the policy of via feed_dict
        into 'state_input' and fetch 'probabilities' which contains a vector
        containing a number for each action how probable this action is given
        the input state.

        Training: Fit the parameters of our policy network according to the
        data we have observed. Feed the observed states, actions and advantages
        via feed_dict into 'state_input', 'actions_input', 'advantages_input'
        and fetch the trainable variables 'trainable_vars'.
        Note: be sure to fetch the trainable weights, otherwise they won't be
        updated.

        Note: The shapes which are written in the comments refer to gyms
        CartPole-v0.

        :param env: the environment we are trying to master
        :return:
            placeholder variable to input the state into the network,
            placeholder variable to input the actions into the network which
                are used for training the network,
            placeholder variable to input the advantages which are produced
                during an episode for training the network,
            estimated probability for each possible action of our neural
                network for current state,
            the trainable variables of the policy network which are updated
                every time when they are fetched
        """
        with tf.variable_scope("actor"):

            # -------------------- Read Dimensions -------------------------- #

            state_dim = env.observation_space.shape[0]

            action_dim = len(env.action_space)

            act_state_dim = state_dim * action_dim

            # -------------------- Input Variables -------------------------- #

            # During runtime we will feed the state(-vector) to the network
            # using this variable.
            state_input = tf.placeholder("float",
                                            [None, state_dim],
                                            name="state_input")

            # - We have to specify shape of actions so we can call get_shape
            # when calculating g_log_prob below. -
            # During runtime we will use this variable to input all the actions
            # which appeared during our episode into our policy network.
            # The amount of action during an episode is a fixed value which has
            # been predefined by the user. Each action is displayed by a
            # one-hot array. All entries are 0, except the action that was
            # taken, which has a 1. This action was chosen stochastically
            # regarding the probabilites of the policy network.
            actions_input = tf.placeholder("float",
                                              [env.time_steps, action_dim],
                                              name="actions_input")


            # Placeholder with just 1 dimension which is dynamic
            # We use it to feed the advantages of our episode to the network.
            # The size of the tensor is determined by the number of steps the
            # agent executes during one episode run.
            advantages_input = tf.placeholder("float", [None, ],
                                                 name="advantages_input")

            # ------------------------ Weights ------------------------------ #

            weights = tf.get_variable("weights", [state_dim, action_dim])

            # ------------------------ Network ------------------------------ #

            # This is our network. It is simple, linear
            # and has just 1 weight tensor.
            linear = tf.matmul(state_input, weights)

            # Softmax function: sum(probabilities) = 1
            probabilities = tf.nn.softmax(linear, name="probabilities")

            # ------------------- Trainable Vars ---------------------------- #

            # Returns a list which only contains weights, as it is our only
            # trainable variable: [tf.Variable with shape =(4, 2)]
            trainable_vars = tf.trainable_variables()

            # ------------------------ π(a|s) ------------------------------- #
            # Calculate the probability of the chosen action given the state

            # We multiply probabilities, which has in every row the
            # probabilites for every possible action, elementwise with the
            # actions_input. Because actions_input is a one-hot array, which
            # only has a 1 at the chosen action, we end up with an array which
            # has in every row just one probability number.
            # Results in shape (200, 2)
            action_prob = tf.multiply(probabilities, actions_input)

            # Now we sum up each row  to get rid of the 0s.
            # This means we end up with a tensor which  has just 1 dimension
            # with "env.time_steps" elements. For every step we took in our
            # env, we now have the probability of the action, that we took.
            # Results in shape (200,)
            action_prob = tf.reduce_sum(action_prob, axis=[1])

            # ----------------------- log(π(a|s)) --------------------------- #

            action_log_prob = tf.log(action_prob)

            # ------------------- ∇_θ log(π(a|s)) --------------------------- #
            # Calculate the gradient of the log probability at each point in
            # time.

            # Flattening, Results in shape (200,)
            action_log_prob_flat = tf.reshape(action_log_prob, (-1,))

            # NOTE: doing this because tf.gradients returns a summed version
            # Take the gradient of each action w.r.t. the trainable weights
            # Results in shape (200, 4, 2): List with 200 tensors of (4, 2)
            num_of_steps = action_log_prob.get_shape()[0]

            g_log_prob = [tf.gradients(
                action_log_prob_flat[i],
                trainable_vars)[0] for i in range(num_of_steps)]

            # Results in shape (200, 4, 2)
            g_log_prob = tf.stack(g_log_prob)

            # Results in shape (200, 8, 1)
            g_log_prob = tf.reshape(g_log_prob,
                                    (env.time_steps, act_state_dim, 1))

            # ---------------------- ∇_θ J(θ) ------------------------------- #

            # Calculate the gradient of the cost function by multiplying
            # the log derivatives of the policy by the advantage function:
            # E[∇_θ log(π(a|s)) A(s,a)]. The expectation E will be taken if we
            # do it for all the (s,a) which we observe and sum it together.

            # The Advantage is currently calculated with the total discounted
            # reward minus the V value which has been estimated by our critic
            # network.
            # Restuls in shape (200, 1, 1)
            adv_reshaped = tf.reshape(advantages_input, (env.time_steps, 1, 1))

            # Each advantage of each time step is multiplied by each partial
            # derivative which we have calculated for that time step.
            # Results in shape (200, 8, 1)
            grad_j = tf.multiply(g_log_prob, adv_reshaped)

            # Get the mean (sum over time and divide by 1/time steps) to get
            # the expectation E. Results in shape (8, 1).
            grad_j = tf.reduce_sum(grad_j, reduction_indices=[0])
            grad_j = 1.00 / env.time_steps * grad_j

            # --------------- Fischer Information Matrix -------------------- #

            # Calculate the Fischer information matrix for every time step.
            # [∇_θ log(π(a|s)) ∇_θ log(π(a|s))^T] ∀ t ∈ time-steps
            # Results in shape (200, 8, 8)
            x_times_x_fct = lambda x: tf.matmul(x, tf.transpose(x))
            fisher = tf.map_fn(x_times_x_fct, g_log_prob)

            # Get the mean (sum over time and divide by 1/time steps) to get
            # the expectation E. Results in shape (8, 8).
            fisher = tf.reduce_sum(fisher, reduction_indices=[0])
            fisher = 1.0 / env.time_steps * fisher

            # Result: fisher = E[∇_θ log(π(a|s)) ∇_θ log(π(a|s))^T]

            # ------------------ Invers Fisher Matrix ----------------------- #

            # We calculate the inverse fisher matrix by using SVD, because
            # we want to clip out small eigenvalues (< 1e-6) which are
            # created as rounding errors. This is necessary because if we take
            # the inverse of a small rounding error, the eigenvalue would be
            # (incorrectly) huge!

            # For this we use Singular Value Decomposition. For the paper and
            # an example, look here: https://doi.org/10.1137/0702016

            s_mat, u_mat, v_mat = tf.svd(fisher)

            lower_bound = tf.reduce_max(s_mat) * 1e-6
            s_inv = tf.divide(1.0, s_mat)

            # If the element in 's' is smaller than the lower bound, we
            # write a 0, otherwise we take the number we calculated as inverse.
            s_inv = tf.where(s_mat < lower_bound, tf.zeros_like(s_mat), s_inv)

            s_inv = tf.diag(s_inv)
            fisher_inv = tf.matmul(s_inv, tf.transpose(u_mat))
            fisher_inv = tf.matmul(v_mat, fisher_inv)

            # --------------------- δθ = Policy Update ---------------------- #
            # We calculate the natural gradient policy update:
            # δθ = α x inverse(fisher) x ∇_θ J(θ)

            # Calculate natural policy gradient ascent update
            fisher_inv_grad_j = tf.matmul(fisher_inv, grad_j)

            # Calculate the learning rate
            # We use the fischer inverse to prevent huge weight changes
            learn_rate = tf.sqrt(tf.divide(
                env.learning_rate_actor,
                tf.matmul(tf.transpose(grad_j), fisher_inv_grad_j)))

            # Multiply natural gradient by a learning rate
            update = tf.multiply(learn_rate, fisher_inv_grad_j)

            # Reshape to (2, 4) because our weight tensor has this shape
            update = tf.reshape(update, (state_dim, action_dim))

            # Update trainable parameters which in our case is just one tensor
            # NOTE: Whenever trainable_vars is fetched they're also updated
            trainable_vars[0] = tf.assign_add(trainable_vars[0], update)

            return state_input, actions_input, advantages_input, \
                probabilities, trainable_vars
