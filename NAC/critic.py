#!/usr/bin/env python

import tensorflow as tf
import numpy as np

__author__ = "Maximilian A. Gehrke, Yannik P. Frisch, Tabea A. Wilke"
__data__ = "14.03.2019"
__copyright__ = "Copyright (C) 2019 Max Gehrke, Yannik Frisch, Tabea Wilke"
__credits__ = ["Maximilian A. Gehrke", "Yannik P. Frisch", "Tabea A. Wilke"]
__license__ = "GPL"
__version__ = "1.0"
__status__ = "Development"


class Critic:

    def __init__(self, env, state_input, true_vf_input,
                 output, optimizer, loss):
        """
        Construct the critic of our natural actor critic algorithm.

        :param env: a MyEnvironment instance
        :param state_input: a tensorflow placeholder which we will feed the
            states of our batch
        :param true_vf_input: a tensorflow placeholder which we will feed the
            discounted return of each state of our batch
        :param output: a tensorflow object which contains the value function
            prediction of the critic if we feed it an observation
        :param optimizer: a tensorflow object which contains the Adam
            optimizer of the critic. We need to call it to update the critic.
        :param loss: a tensorflo object which contains the loss when we
            the critic.
        """

        self.env = env
        self.state_input = state_input
        self.true_vf_input = true_vf_input
        self.output = output
        self.optimizer = optimizer
        self.loss = loss

    def update(self, sess, batch_states, batch_discounted_returns):
        """
        Update the weights of our critic.

        :param sess: a tensorflow session
        :param batch_states: an array containing all the states of a batch
        :param batch_discounted_returns: an array containing all the
            discounted rewards of a batch
        :return: None
        """
        returns_vector = np.expand_dims(batch_discounted_returns, axis=1)
        sess.run(self.optimizer,
                 feed_dict={self.state_input: batch_states,
                            self.true_vf_input: returns_vector})

    def estimate(self, sess, observation):
        """
        Estimate the value of a given state/observation using the critic.

        :param sess: a tensorflow session
        :param observation: an observation/state of our environment
        :return: the estimated value of the given state
        """

        observation_vector = np.expand_dims(observation, axis=0)
        state_value = sess.run(
            self.output,
            feed_dict={self.state_input: observation_vector}
        )[0][0]

        return state_value

    @staticmethod
    def create_value_net(env):
        """
        Function approximation of the value function for states in our
        environment. We use a neural network with the following structure:
            Input Layer: number of nodes = dim of state, fully connected.
            Hidden Layer: fully connected, ReLu activation.
            Output Layer: 1 node.
        Finally, we use an AdamOptimizer to train our NN by Gradient Descent.

        :param env: the environment we are working with
        :return:
            estimated value of our neural network,
            placeholder variable to input the state into the network,
            placeholder variable to input the true value function value for the
                above state,
            the adam optimizer object,
            the loss between the true value and the estimated value for the state
        """
        with tf.variable_scope("critic"):
            # Get the state size to get the number of input nodes
            state_size = env.observation_space.shape[0]

            # Input layer, hidden dense layer,
            # bias b1 & ReLu activation
            state_input = tf.placeholder("float", [None, state_size],
                                         name="state_input")
            w1 = tf.get_variable("w1", [state_size, env.hidden_layer_critic])
            b1 = tf.get_variable("b1", [env.hidden_layer_critic])
            h1 = tf.nn.relu(tf.matmul(state_input, w1) + b1)

            # Output times 2nd weights plus 2nd bias
            w2 = tf.get_variable("w2", [env.hidden_layer_critic, 1])
            b2 = tf.get_variable("b2", [1])
            output = tf.add(tf.matmul(h1, w2), b2, name="output")

            # During runtime this value will hold the true value
            # (discounted return) of the value function which we will use to
            # adjust our NN accordingly.
            true_vf_input = tf.placeholder("float", [None, 1],
                                           name="true_vf_input")

            # Minimize the difference between predicted and actual output
            diffs = output - true_vf_input
            loss = tf.nn.l2_loss(diffs, name="loss")  # sum (diffs ** 2) / 2

            # Computes the gradients of the network and applies them again so
            # the 'loss' value will be minimized. This is done via the Adam
            # algorithm.
            optimizer = tf.train.AdamOptimizer(env.learning_rate_critic)
            optimizer = optimizer.minimize(loss)

            tf.add_to_collection("optimizer", optimizer)

            return state_input, true_vf_input, output, optimizer, loss
