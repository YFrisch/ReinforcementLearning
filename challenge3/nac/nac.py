import numpy as np
import random
import gym
import quanser_robots
import warnings


def run_batch(env, actor, value_grad, sess, num_traj, printing=False):
    """
    TODO
    :param env:
    :param actor:
    :param value_grad:
    :param sess:
    :param num_traj:
    :param printing:
    :return:
    """

    continuous = env.continuous

    # Unpack the policy network (generates control policy)
    (pl_state, pl_actions, pl_advantages,
        pl_probabilities, pl_train_vars) = actor.get_net_variables()

    # Unpack the value network (estimates expected reward)
    (vfa_state_input, vfa_true_vf_input,
        vfa_nn_output, vfa_optimizer, vfa_loss) = value_grad

    # set up the environment
    observation = env.reset()

    traj_reward = 0.0
    traj_transitions = []

    batch_traj_rewards = []
    batch_states = []
    batch_actions = []
    batch_advantages = []
    batch_discounted_returns = []

    n_trajectories = 0
    n_timesteps = env.time_steps

    for t in range(n_timesteps):

        # I think sometimes we have a zero in the observations
        # and we somehow divide while calculating the probs
        observation = \
            [0.00001 if np.abs(x) < 0.00001 else x for x in observation]

        if env.name == 'Qube-v0':
            for rr in range(4):
                ob = observation[rr]
                if ob > 0.999:
                    observation[rr] = 0.999
                elif ob < -0.999:
                    observation[rr] = -0.999

        # Expand state by one dimension
        # Before = [ ... , ... , ... ], After = [[ ... , ... , ... ]]
        obs_vector = np.expand_dims(observation, axis=0)

        print("({}) OBS:{}".format(t, obs_vector), end='') if printing else ...

        # ------------------- PREDICT ACTION -------------------------------- #
        if continuous:
            action = sess.run(
                pl_probabilities,
                feed_dict={pl_state: obs_vector})

            batch_actions.append(action)

        else:
            action, action_i = actor.get_action(sess, observation)

            # Make one-hot action array
            action_array = np.zeros(len(env.action_space))
            action_array[action_i] = 1
            batch_actions.append(action_array)
            print(", ACTION ARRAY: ", action_array, end='') if printing else ...

        print(", ACTION:", action, end='') if printing else ...

        # Record transition
        batch_states.append(observation)
        old_observation = observation
        # print("B4:", t, action)
        observation, reward, done, info = env.step(action)
        print("After:", t, observation, reward, done)

        traj_transitions.append((old_observation, action, reward))
        traj_reward += reward

        # -------------------- END OF TRAJECTORY ---------------------------- #

        # If env = done or we collected our desired number of steps
        if done or t == n_timesteps - 1:
            # TODO: Save computation time by calculating G_t backwards
            for i_trans, trans in enumerate(traj_transitions):
                obs, action, reward = trans
                obs_vector = np.expand_dims(obs, axis=0)

                # --------- Discounted monte-carlo return (G_t) ------------- #

                discounted_return = 0
                future_transitions_n = len(traj_transitions) - i_trans
                decrease = 1
                for p in range(future_transitions_n):
                    discounted_return += traj_transitions[p + i_trans][2] * decrease
                    decrease = decrease * env.mc_discount_factor

                # save disc reward to update critic params in its direction
                batch_discounted_returns.append(discounted_return)

                # --------- Get the value V from our Critic ----------------- #

                # The estimated return the critic expects the state to have
                critic_value = sess.run(
                    vfa_nn_output,
                    feed_dict={vfa_state_input: obs_vector}
                )[0][0]

                # --------------------- ADVANTAGE --------------------------- #

                # How much better did we do compared to the critic expectation
                batch_advantages.append(discounted_return - critic_value)

            # How many trajectories have been executed in this batch
            n_trajectories += 1
            # Save current reward of trajectory
            batch_traj_rewards.append(traj_reward)

            # ------------------- RESET VARIABLES --------------------------- #
            traj_reward = 0.0
            traj_transitions = []

            if done:
                # reset the environment, if we still have steps left
                observation = env.reset()
            else:
                # if we have no steps left, close environment
                env.close()

    print("\n\n\n") if printing else ...
    print('Update {} with {} trajectories with rewards of: {}'
          .format(num_traj, len(batch_traj_rewards), batch_traj_rewards))

    # ----------------- UPDATE VALUE NETWORK -------------------------------- #
    batch_disc_returns_vec = np.expand_dims(batch_discounted_returns, axis=1)
    sess.run(vfa_optimizer,
             feed_dict={vfa_state_input: batch_states,
                        vfa_true_vf_input: batch_disc_returns_vec})

    # ---------------- UPDATE POLICY NETWORK -------------------------------- #
    print("Policy update:", np.asarray(batch_states).shape,
          np.asarray(batch_advantages).shape,
          np.asarray(batch_actions).shape) if printing else ...

    actor.update(sess, batch_states, batch_actions, batch_advantages)

    return batch_traj_rewards, n_trajectories
