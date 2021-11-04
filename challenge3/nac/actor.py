import tensorflow as tf

import gym
import quanser_robots
import numpy as np
import sys
import random


class Actor:
    def __init__(self, env, printing=False):

        self.env = env
        self.pl_state_input, self.pl_actions_input, self.pl_advantages_input, \
            self.pl_probabilities, self.pl_train_vars \
            = create_policy_net(env, printing)

    def update(self, sess, batch_states, batch_actions, batch_advantages):
        sess.run(self.pl_train_vars,
                 feed_dict={self.pl_state_input: batch_states,
                            self.pl_actions_input: batch_actions,
                            self.pl_advantages_input: batch_advantages})

    def get_action(self, sess, observation):

        # Get probabilites of actions to take
        obs_vector = np.expand_dims(observation, axis=0)
        probs = sess.run(
            self.pl_probabilities,
            feed_dict={self.pl_state_input: obs_vector})

        # Stochastically generate an action using the policy output probs
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
        This method returns the variables of our policy network. This method
        shall be used for development purposes only. If development has
        finished, a new method imbedding the desired functionality shall be
        created.

        :return: placeholder state variable,
            placeholder action variable,
            placeholder advantages variable,
            network output variable (= probabilites of actions),
            trainable weights variable
        """
        return self.pl_state_input, self.pl_actions_input, \
            self.pl_advantages_input, self.pl_probabilities, \
            self.pl_train_vars


def create_policy_net(env, printing=False):
    """
    Neural Network to approximate our policy.

    Estimating: We want to know the probabilities the network outputs for a
    state. Just feed the state you want to know the policy of via feed_dict
    into 'pl_state_input' and fetch 'pl_probabilities' which contains a vector
    containing a number for each action how probable this action is given the
    input state.

    Training: Fit the parameters of our policy network according to the data
    we have observed. Feed the observed states, actions and advantages via
    feed_dict into 'pl_state_input', 'pl_actions_input', 'pl_advantages_input'
    and fetch the trainable variables 'pl_train_vars'.
    Note: be sure to fetch the trainable weights, otherwise they won't be
    updated. Todo: correct?

    :param env: the environment we are trying to master
    :return:
        placeholder variable to input the state into the network,
        placeholder variable to input the actions into the network which are
            used for training the network,
        placeholder variable to input the advantages which are produced during
            during an episode for training the network,
        estimated probability for each possible action of our neural network
            for current state,
        the trainable variables of the policy network which are updated every
            time when they are fetched
    """
    with tf.variable_scope("policy"):

        # ------------------ Read Dimensions ---------------------- #

        # TODO: Actions can have several dimensions
        state_dim = env.observation_space.shape[0]
        if env.continuous:
            action_dim = env.action_dimensions
        else:
            action_dim = len(env.action_space)

        act_state_dim = state_dim * action_dim

        # -------------------- Input Variables -------------------- #

        # During runtime we will feed the state(-vector) to the network
        # using this variable.
        pl_state_input = tf.placeholder("float",
                                        [None, state_dim],
                                        name="pl_state_input")

        # - We have to specify shape of actions so we can call get_shape
        # when calculating g_log_prob below. -
        # During runtime we will use this variable to input all the actions
        # which appeared during our episode into our policy network.
        # The amount of action during an episode is a fixed value which has
        # been predefined by the user. Each action is displayed by a one-hot
        # array. All entries are 0, except the action that was taken, which
        # has a 1. This action was chosen stochastically regarding the
        # probabilites of the policy network.
        pl_actions_input = tf.placeholder("float",
                                          [env.time_steps, action_dim],
                                          name="pl_actions_input")


        # Placeholder with just 1 dimension which is dynamic
        # We use it to feed the advantages of our episode to the network. The
        # size of the tensor is determined by the number of steps the agent
        # executes during one episode run.
        pl_advantages_input = tf.placeholder("float", [None, ],
                                             name="pl_advantages_input")

        # ------------------------ Weights ------------------------ #

        if env.complex_policy:

            # Input layer, hidden dense layer (size 10), bias b1 & ReLu activation
            w1 = tf.get_variable("pl_w1", [state_dim, env.hidden_layer_size])
            b1 = tf.get_variable("pl_b1", [env.hidden_layer_size])

            # Output times 2nd weights plus 2nd bias
            w2 = tf.get_variable("pl_w2", [env.hidden_layer_size, action_dim])
            b2 = tf.get_variable("pl_b2", [action_dim])

            weight_dims = [w1.get_shape().as_list(),
                           b1.get_shape().as_list(),
                           w2.get_shape().as_list(),
                           b2.get_shape().as_list()]

            weight_sizes = [np.prod(x) for x in weight_dims]
            weight_indices = np.insert(np.cumsum(weight_sizes)[:-1], 0, 0)
            weight_sizes = [[x, 1] for x in weight_sizes]
            weight_indices = [[x, 0] for x in weight_indices]

            # weight_indices = [(0, 1), (40, 1), (50, 1), (70, 1)]
            # weight_sizes = [(40, 1), (10, 1), (20, 1), (2, 1)]
            #
            # weight_dims = [(4, 10), (10,), (10, 2), (2,)]

            print("\nWEIGHT INDICES:", weight_indices) if printing else ...
            print("WEIGHT SIZES:", weight_sizes) if printing else ...
            print("WEIGHT DIMS:", weight_dims) if printing else ...

        else:
            pl_weights = tf.get_variable("pl_weights", [state_dim, action_dim])

        # ------------------------ Network ------------------------ #

        if env.complex_policy:
            h1 = tf.nn.relu(tf.matmul(pl_state_input, w1) + b1)
            pl_probabilities_t = tf.nn.softmax(tf.matmul(h1, w2) + b2)
            pl_probabilities = tf.Print(pl_probabilities_t, [pl_probabilities_t, w1, w2])

        else:
            # This is our network. It is simple, linear
            # and has just 1 weight tensor.
            linear = tf.matmul(pl_state_input, pl_weights)

            if env.continuous:
                # Tangenz-H: -1 < x 1
                pl_probabilities = tf.nn.tanh(linear)

                bias = [(env.action_space_high[i] + env.action_space_low[i])
                        / 2 for i in range(action_dim)]

                # Scale, so it matches the action range
                # TODO: What if max action = inf

                pl_probabilities = tf.map_fn(
                    lambda x: x - bias * env.action_space_high,
                    pl_probabilities)

            else:
                # Softmax function: sum(probabilities) = 1
                pl_probabilities = tf.nn.softmax(linear)

        # ------------------- Trainable Vars ------------------------ #

        # Returns a list which only contains pl_weights, as it is our only
        # trainable variable: [tf.Variable with shape =(4, 2)]
        pl_train_vars = tf.trainable_variables()
        print("\nSHAPE A:", pl_train_vars) if printing else ...

        # ------------------------ π(a|s) -------------------------- #
        if env.continuous:
            action_prob = tf.map_fn(lambda x: 1, pl_probabilities)
        else:
            # Calculate the probability of the chosen action given the state
            # TODO: just  get the prob of the chosen action or an array with
            #   all probs of possible actions?

            # We multiply probabilities, which has in every row the probabilites
            # for every possible action, elementwise with the actions_input.
            # Because actions_input is a one-hot array, which only has a 1 at the
            # chosen action, we end up with an array which has in every row just
            # one probability number.
            # Results in shape (200, 2)
            action_prob = tf.multiply(pl_probabilities, pl_actions_input)
            print("SHAPE B:", action_prob.shape) if printing else ...

            # Now we sum up each row  to get rid of the 0s.
            # This means we end up with a tensor which  has just 1 dimension with
            # "env.time_steps" elements. For every step we took in our env, we now
            # have the probability of the action, that we took.
            # Results in shape (200,)
            action_prob = tf.reduce_sum(action_prob, axis=[1])
            print("SHAPE C:", action_prob.shape) if printing else ...

        # ----------------------- log(π(a|s)) ----------------------- #

        action_log_prob = tf.log(action_prob)
        print("SHAPE D:", action_log_prob.shape) if printing else ...

        # ------------------- ∇_θ log(π(a|s)) ----------------------- #
        # Calculate the gradient of the log probability at each point in time


        # NOTE: doing this because tf.gradients only returns a summed version
        # TODO: As far as I can tell, this is unnecessary. The array is already
        #   flattened. Maybe when we have high dim actions it will be useful.
        # Results in shape (200,)
        action_log_prob_flat = tf.reshape(action_log_prob, (-1,))


        # TODO: Do we want to take gradient of scalar and all weights
        # Take the gradient of each action w.r.t. the trainable weights
        # Results in shape (200, 4, 2): List with 200 tensors of shape (4, 2)
        num_of_steps = action_log_prob.get_shape()[0]
        if env.complex_policy:
            g_log_prob = [tf.gradients(action_log_prob_flat[i], pl_train_vars)
                        for i in range(num_of_steps)]
            print("SHAPE E:", g_log_prob) if printing else ...

            g_log_prob = [[tf.reshape(y, [-1, 1]) for y in x] for x in g_log_prob]
            print("SHAPE F:", g_log_prob) if printing else ...

            g_log_prob = [tf.concat(x, 0) for x in g_log_prob]
            print("SHAPE G:", g_log_prob) if printing else ...

            g_log_prob = tf.stack(g_log_prob)
            print("SHAPE H:", g_log_prob.shape) if printing else ...


        else:
            g_log_prob = [tf.gradients(action_log_prob_flat[i], pl_train_vars)[0]
                        for i in range(num_of_steps)]

            # Results in shape (200, 4, 2)
            g_log_prob = tf.stack(g_log_prob)

            # Results in shape (200, 8, 1)
            g_log_prob = tf.reshape(g_log_prob, (env.time_steps, act_state_dim, 1))

        # ------------------- ∇_θ J(θ) ----------------------- #

        # Calculate the gradient of the cost function by multiplying
        # the log derivatives of the policy by the advantage function:
        # E[∇_θ log(π(a|s)) A(s,a)]. The expectation E will be taken if we do
        # it for all the (s,a) which we observe and sum it together.

        # The Advantage is currently calculated with the total discounted
        # reward minus the V value which has been estimated by our critic
        # network.
        # Restuls in shape (200, 1, 1)
        adv_reshaped = tf.reshape(pl_advantages_input, (env.time_steps, 1, 1))

        # Each advantage of each time step is multiplied by each partial
        # derivative which we have calculated for that time step.
        # Results in shape (200, 8, 1)
        grad_j = tf.multiply(g_log_prob, adv_reshaped)
        print("SHAPE I:", grad_j.shape) if printing else ...

        # Get the mean (sum over time and divide by 1/time steps) to get the
        # expectation E. Results in shape (8, 1).
        grad_j = tf.reduce_sum(grad_j, reduction_indices=[0])
        grad_j = 1.00 / env.time_steps * grad_j
        print("SHAPE J:", grad_j.shape) if printing else ...

        # --------------- Fischer Information Matrix --------------- #

        # Calculate the Fischer information matrix for every time step.
        # [∇_θ log(π(a|s)) ∇_θ log(π(a|s))^T] ∀ t ∈ time-steps
        # Results in shape (200, 8, 8)
        x_times_xT_fct = lambda x: tf.matmul(x, tf.transpose(x))
        fisher = tf.map_fn(x_times_xT_fct, g_log_prob)
        print("SHAPE K:", fisher.shape) if printing else ...

        # Get the mean (sum over time and divide by 1/time steps) to get the
        # expectation E. Results in shape (8, 8).
        fisher = tf.reduce_sum(fisher, reduction_indices=[0])
        fisher = 1.0 / env.time_steps * fisher
        print("SHAPE L:", fisher.shape) if printing else ...


        # Result: fisher = E[∇_θ log(π(a|s)) ∇_θ log(π(a|s))^T]

        # ------------------------ SVD Clip ------------------------ #

        # Calculate inverse of positive definite clipped F
        # NOTE: have noticed small eigenvalues (1e-10) that are negative,
        # using SVD to clip those out, assuming they're rounding errors
        S, U, V = tf.svd(fisher)

        atol = tf.reduce_max(S) * 1e-6
        S_inv = tf.divide(1.0, S)

        # If the element in S(!) is smaller than the lower bound 'atol', we
        # write a 0, otherwise we take the number we calculated as inverse.
        S_inv = tf.where(S < atol, tf.zeros_like(S), S_inv)
        S_inv = tf.diag(S_inv)
        fisher_inv = tf.matmul(S_inv, tf.transpose(U))
        fisher_inv = tf.matmul(V, fisher_inv)

        # --------------------- δθ = Policy Update --------------------- #
        # We calculate the natural gradient policy update:
        # δθ = α x inverse(fisher) x ∇_θ J(θ)

        # Calculate natural policy gradient ascent update
        fisher_inv_grad_j = tf.matmul(fisher_inv, grad_j)
        print("SHAPE M:", fisher_inv_grad_j.shape) if printing else ...


        # TODO: How does learning rate changes change the output?
        # Calculate a learning rate normalized such that a constant change
        # in the output control policy is achieved each update, preventing
        # any parameter changes that hugely change the output
        learn_rate = tf.sqrt(tf.divide(
            env.learning_rate_actor,
            tf.matmul(tf.transpose(grad_j), fisher_inv_grad_j)
        ))

        # Multiply natural gradient by a learning rate
        pl_update = tf.multiply(learn_rate, fisher_inv_grad_j)

        if env.complex_policy:

            for ind in range(len(pl_train_vars)):
                w_ind = weight_indices[ind]
                w_siz = weight_sizes[ind]
                w_dim = weight_dims[ind]

                print("({}) WEIGHT Index: {}, Size: {}, Dims: {}:"
                      .format(ind, w_ind, w_siz, w_dim)) if printing else ...

                update_tensor = tf.slice(pl_update, w_ind, w_siz)
                print("SHAPE N:", update_tensor.shape) if printing else ...

                update_tensor = tf.reshape(update_tensor, w_dim)
                print("SHAPE O:", update_tensor.shape) if printing else ...

                pl_train_vars[ind] = tf.assign_add(pl_train_vars[ind], update_tensor)


        else:
            # Reshape to (2, 4) because our weight tensor has this shape
            pl_update = tf.reshape(pl_update, (state_dim, action_dim))

            # Update trainable parameters which in our case is just one tensor
            # NOTE: Whenever pl_train_vars is fetched they're also updated
            pl_train_vars[0] = tf.assign_add(pl_train_vars[0], pl_update)

        return pl_state_input, pl_actions_input, pl_advantages_input, \
            pl_probabilities, pl_train_vars
