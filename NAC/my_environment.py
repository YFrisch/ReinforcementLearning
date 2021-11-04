#!/usr/bin/env python

import random
import numpy as np
import gym
import quanser_robots
import warnings
import datetime
import itertools

__author__ = "Maximilian A. Gehrke, Yannik P. Frisch, Tabea A. Wilke"
__data__ = "14.03.2019"
__copyright__ = "Copyright (C) 2019 Max Gehrke, Yannik Frisch, Tabea Wilke"
__credits__ = ["Maximilian A. Gehrke", "Yannik P. Frisch", "Tabea A. Wilke"]
__license__ = "GPL"
__version__ = "1.0"
__status__ = "Development"


class MyEnvironment():
    def __init__(self, env_details):
        """
        Construct an environment class which is adapted to our algorithm.
        MyEnviornment shall contain all important variables and constants which
        will be accessed by other classes.

        The idea of this class is that no matter which environment we want to
        solve with our natural actor critic algorithm, we just need to adjust
        this class.

        The constructor creates the underlying environment, the action space
        and the observation space and prints important information about the
        environment to the console.

        :param env_details: this list contains the name of the enviornment
            we want to solve and all its hyperparameters
        """

        self.name = env_details[0]
        self.action_form = env_details[1]
        self.discretization = env_details[2]
        self.time_steps = env_details[3]  # batch size, steps btw. updates
        self.num_of_updates = env_details[4]
        self.mc_discount_factor = env_details[5]
        self.learning_rate_actor = env_details[6]
        self.learning_rate_critic = env_details[7]
        self.hidden_layer_critic = env_details[8]

        self.network_generation_time = 0
        self.network_training_time = 0

        # -------------- CREATE FOLDER NAME FOR SAVING ---------------------- #

        # Get current time
        save_time = datetime.datetime.now()

        self.save_folder = "{}/{}/{}-{}-{}_{}-{}-{}" \
            .format('data', self.name, save_time.year, save_time.month,
                    save_time.day, save_time.hour, save_time.minute,
                    save_time.second)

        # ------------------  CREATE GYM ENVIRONMENT ------------------------ #

        # Ignoring PkgResourcesDeprecationWarning: Parameters deprecated.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.env = gym.make(self.name)

        # --------------------- OBSERVATION SPACE --------------------------- #

        self.observation_space = self.env.observation_space
        self.observation_space.high = self.env.observation_space.high
        self.observation_space.low = self.env.observation_space.low

        print("\tObservation space high: {}"
              .format(self.observation_space.high))
        print("\tObservation space low : {}"
              .format(self.observation_space.low))
        print("\tOriginal action space object type:",
              type(self.env.action_space))

        # ------------------------ ACTION SPACE ----------------------------- #

        # Need conditions for different action structure of different classes
        if type(self.discretization[0]) == list:
            self.action_space = np.asarray(self.discretization[0])
            self.action_dimensions = np.shape(self.action_space.shape)[0]
            self.action_space_n = self.action_space.shape[0]

        elif type(self.env.action_space) is gym.spaces.discrete.Discrete:
            assert env_details[1] == 'discrete'
            assert len(set(env_details[2])) == 1 and 0 in env_details[2]
            self.action_space = np.arange(self.env.action_space.n)
            self.action_space = [[x] for x in self.action_space]
            self.action_space = np.asarray(self.action_space)
            self.action_space_n = self.env.action_space.n
            self.action_dimensions = 1

        elif type(self.env.action_space) in \
                [gym.spaces.box.Box, quanser_robots.common.LabeledBox]:
            assert env_details[1] == 'continuous'
            self.action_dimensions = len(self.env.action_space.low)
            self.action_space = []
            for i in range(self.action_dimensions):
                actions_in_dim = np.linspace(self.env.action_space.low[i],
                                                self.env.action_space.high[i],
                                                self.discretization[i])
                if i == 0:
                    self.action_space = actions_in_dim
                else:
                    self.action_space = list(itertools.product(
                        self.action_space, actions_in_dim))
            if self.action_dimensions == 1:
                self.action_space = [[x] for x in self.action_space]
            self.action_space = np.asarray(self.action_space)
            self.action_space_n = np.prod(self.discretization)

        else:
            raise ValueError("Env Action Space should be of type Discrete "
                             "or Box, but is of Type {}."
                             .format(type(self.env.action_space)))

        self.env.action_space = self.action_space
        self.action_space_high = \
            np.asarray([np.max(self.action_space, axis=0)])
        self.action_space_low = np.asarray([np.min(self.action_space, axis=0)])
        print("\tAction space high: {}".format(self.action_space_high))
        print("\taction space low : {}".format(self.action_space_low))
        print("\tAction space: {}".format(self.action_space.tolist()))

        # Print reward range (if (-inf, inf) it is not set in the environment)
        print("\tReward range: {}".format(self.env.reward_range))

    def step(self, action):
        """
        Take the given action in the environment.

        :param action: the action we want to execute in the environment
        :return: observation, reward, done, info
        """

        # Try/Except: Some env need the action in an array, others (1D) don't
        try:
            action_tmp = np.array(action[0])
            observation, reward, done, info = self.env.step(action_tmp)
        except AssertionError:
            observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    def reset(self):
        """Reset the environment and return the start state."""
        return self.env.reset()

    def close(self):
        """Close the environment"""
        self.env.close()

    def render(self):
        "Render the environment"
        self.env.render()

    def action_space_sample(self):
        """Sample a random action from the action space."""
        return random.choice(self.action_space)
