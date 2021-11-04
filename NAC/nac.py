#!/usr/bin/env python

import numpy as np

__author__ = "Maximilian A. Gehrke, Yannik P. Frisch, Tabea A. Wilke"
__data__ = "14.03.2019"
__copyright__ = "Copyright (C) 2019 Max Gehrke, Yannik Frisch, Tabea Wilke"
__credits__ = ["Maximilian A. Gehrke", "Yannik P. Frisch", "Tabea A. Wilke"]
__license__ = "GPL"
__version__ = "1.0"
__status__ = "Development"


class NAC:
    def __init__(self, env, actor, critic):
        """
        This class records transitions by executing steps in the environment,
        which uses the predicted actions of our actor neural network.

        Afterwards, it calculates the discounted return and the critic's value
        function prediction for every state and subtracts the latter from
        the first to get the advantages.

        Finally, it updates the critic by using the discounted returns and the
        actor by using the advantages.

        :param env: a my_environment object, which embodies the gym or
            quanser_robots environment that we want to solve
        :param actor: an actor object, which embodies a policy neural network
        :param critic: a critic object, which embodies a value function neural
            network
        """
        self.env = env
        self.actor = actor
        self.critic = critic

    def run_batch(self, sess):
        """
        Execute one batch worth of transitions and update the actor and critic
        parameters.

        :param sess: the active tensorflow session
        :return: a list of the summed rewards for each trajectory
        """

        # Reset the environment and get start state
        observation = self.env.reset()

        # Variables saving data for the current trajectory
        traj_reward = 0.0
        traj_transitions = []

        # Variables saving data for the complete batch
        batch_traj_rewards = []
        batch_states = []
        batch_actions = []
        batch_advantages = []
        batch_discounted_returns = []

        for t in range(self.env.time_steps):
            # Some environments need preprocessing
            observation = self.preprocess_obs(observation)

            # ------------------- PREDICT ACTION ---------------------------- #

            # Get the action with the highest probability w.r.t our actor
            action, action_i = self.actor.get_action(sess, observation)

            # Make one-hot action array
            action_array = np.zeros(len(self.env.action_space))
            action_array[action_i] = 1
            batch_actions.append(action_array)

            # --------------- TAKE A STEP IN THE ENV ------------------------ #

            old_observation = observation
            observation, reward, done, _ = self.env.step(action)

            reward = reward

            # Record state/transition
            batch_states.append(old_observation)
            traj_transitions.append((old_observation, action, reward))
            traj_reward += reward

            # -------------------- END OF TRAJECTORY ------------------------ #

            # If env = done or we collected our desired number of steps
            if done or t == self.env.time_steps - 1:

                discounted_return = 0.0
                traj_advantages = []
                traj_discounted_returns = []

                # Calculate in reverse order, because it's faster
                for trans_i in reversed(range(len(traj_transitions))):
                    obs, action, reward = traj_transitions[trans_i]

                    # ------- Discounted monte-carlo return (G_t) ----------- #

                    discounted_return = discounted_return * \
                                        self.env.mc_discount_factor + reward

                    # Save disc reward to update critic params in its direction
                    traj_discounted_returns.insert(0, discounted_return)

                    # ------------------- ADVANTAGE ------------------------- #

                    # Get the value V from our Critic
                    critic_value = self.critic.estimate(sess, obs)

                    # Save advantages to update actor params in its direction
                    traj_advantages.insert(0, discounted_return - critic_value)

                # ----------------- SAVE VARIABLES -------------------------- #

                batch_discounted_returns.extend(traj_discounted_returns)
                batch_advantages.extend(traj_advantages)
                batch_traj_rewards.append(traj_reward)

                # ----------------- RESET VARIABLES ------------------------- #
                traj_reward = 0.0
                traj_transitions = []

                if done:
                    # Reset environment, if we still have steps left in batch
                    observation = self.env.reset()
                else:
                    # If we have no steps left, close environment
                    self.env.close()

        # ------------------ UPDATE NETWORKS -------------------------------- #

        self.critic.update(sess, batch_states, batch_discounted_returns)
        self.actor.update(sess, batch_states, batch_actions, batch_advantages)

        return batch_traj_rewards

    def preprocess_obs(self, obs):
        """
        We need to preprocess our observations for two reasons
        1. We do not want to have zeros, because it is not feasible with our
        way of calculating the fisher inverse (dividing by zero).
        2. Some environments have some constraints or function better if the
        state (specific w.r.t. the environment) is clipped.

        :param obs: the observation we want to preprocess
        :return: the preprocessed observation
        """

        obs = [0.00001 if np.abs(x) < 0.00001 else x for x in obs]

        if self.env.name == 'Qube-v0':
            for rr in range(4):
                obs_value = obs[rr]
                if obs_value > 0.999:
                    obs[rr] = 0.999
                elif obs_value < -0.999:
                    obs[rr] = -0.999

        return obs

