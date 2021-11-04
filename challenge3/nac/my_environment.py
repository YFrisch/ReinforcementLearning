import random
import numpy as np
import sys
import gym
import quanser_robots
from quanser_robots import GentlyTerminating
import warnings
import datetime
import os
import itertools


class MyEnvironment(gym.Space):
    def __init__(self, env_details, continuous, complex_policy_net,
                 hidden_layer_size):
        gym.Space.__init__(self, (), np.float)

        self.name = env_details[0]
        self.action_form = env_details[1]
        self.discretization = env_details[2]
        self.time_steps = env_details[3]  # between weight updates
        self.num_of_updates = env_details[4]
        self.mc_discount_factor = env_details[5]
        self.learning_rate_actor = env_details[6]
        self.learning_rate_critic = env_details[7]


        self.continuous = continuous
        self.complex_policy = complex_policy_net
        self.hidden_layer_size = hidden_layer_size

        self.network_generation_time = 0

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
            # self.env = GentlyTerminating(gym.make(env_name))  # doesnt work

        # self.env = gym.wrappers.Monitor(
        #     env=self.env,
        #     directory=self.save_folder,
        #     force=True,
        #     video_callable=True)

        # OBSERVATION SPACE
        self.observation_space = self.env.observation_space
        self.observation_space.high = self.env.observation_space.high
        self.observation_space.low = self.env.observation_space.low

        print("\tObservation space high: {}".format(self.observation_space.high))
        print("\tObservation space low : {}".format(self.observation_space.low))

        print("\tOriginal action space object type:", type(self.env.action_space))

        # ACTION SPACE
        # Need conditions for different action structure of different classes
        if type(self.env.action_space) is gym.spaces.discrete.Discrete:
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
                    self.action_space = list(itertools.product(self.action_space, actions_in_dim))
            if self.action_dimensions == 1:
                self.action_space = [[x] for x in self.action_space]
            self.action_space = np.asarray(self.action_space)
            self.action_space_n = np.prod(self.discretization)
        else:
            raise ValueError("Env Action Space should be of type Discrete "
                             "or Box, but is of Type {}."
                             .format(type(self.env.action_space)))

        self.env.action_space = self.action_space
        self.action_space_high = np.asarray([np.max(self.action_space, axis=0)])
        self.action_space_low = np.asarray([np.min(self.action_space, axis=0)])
        print("\tAction space high: {}".format(self.action_space_high))
        print("\taction space low : {}".format(self.action_space_low))
        print("\tAction space: {}".format(self.action_space.tolist()))

    def step(self, action):
        # Take the action in the environment
        # Try/Except: Some env need action in array, others (1D) don't
        try:
            action_tmp = np.array(action[0])
            observation, reward, done, info = self.env.step(action_tmp)
        except AssertionError:
            observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    def sample_env_action(self):
        return self.env.action_space.sample()

    def reset(self):
        return self.env.reset()

    def close(self):
        self.env.close()

    def render(self):
        self.env.render()

    def action_space_contains(self, x):
        """
        :param x: action space values (as array or similar)
        :return: index of the given actions (as list)
        """
        indices = []
        for i in x:
            indices.append(np.where(self.action_space == i)[0][0])
        return indices

    def action_space_sample(self):
        return random.choice(self.action_space)
