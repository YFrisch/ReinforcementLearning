import tensorflow as tf




def value_gradient(env):
    """
    Function approximation of the value function for states in our environment.
    We use a neural network with the following structure:
        Input Layer: number of nodes = dim of state, fully connected.
        Hidden Layer: 10 nodes, fully connected, ReLu activation.
        Output Layer: 1 node.
    Finally, we use an AdamOptimizer to train our NN by Gradient Descent.

    :param env: the environment we are working with
    :param adam_learn_rate: the learning rate which shall be used in the Adam
        optimizer.
    :return:
        estimated value of our neural network,
        placeholder variable to input the state into the network,
        placeholder variable to input the true value function value for the
            above state,
        the adam optimizer object,
        the loss between the true value and the estimated value for the state
    """
    with tf.variable_scope("value"):

        # Get the state size to get the number of input nodes
        state_size = env.observation_space.shape[0]

        # TODO: Why None and not 1 to construct row vector or just vector
        #   which will be transposed later. Do we do it for vector inputs?
        # Input layer, hidden dense layer (size 10), bias b1 & ReLu activation
        vfa_state_input = tf.placeholder("float", [None, state_size])
        w1 = tf.get_variable("w1", [state_size, 10])
        b1 = tf.get_variable("b1", [10])
        h1 = tf.nn.relu(tf.matmul(vfa_state_input, w1) + b1)

        # Output times 2nd weights plus 2nd bias
        w2 = tf.get_variable("w2", [10, 1])
        b2 = tf.get_variable("b2", [1])
        vfa_nn_output = tf.matmul(h1, w2) + b2

        # During runtime this value will hold the true value
        # (discounted return) of the value function which we will use to
        # adjust our NN accordingly.
        vfa_true_vf_input = tf.placeholder("float", [None, 1])

        # Minimize the difference between predicted and actual output
        diffs = vfa_nn_output - vfa_true_vf_input
        vfa_loss = tf.nn.l2_loss(diffs)  # sum (diffs ** 2) / 2

        # Computes the gradients of the network and applies them again so the
        # 'loss' value will be minimized. This is done via the Adam algorithm.
        vfa_optimizer = tf.train.\
            AdamOptimizer(env.learning_rate_critic).minimize(vfa_loss)

        return vfa_state_input, vfa_true_vf_input, \
            vfa_nn_output, vfa_optimizer, vfa_loss
