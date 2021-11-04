#!/usr/bin/env python
import os

__author__ = "Maximilian A. Gehrke, Yannik P. Frisch, Tabea A. Wilke"
__data__ = "14.03.2019"
__copyright__ = "Copyright (C) 2019 Max Gehrke, Yannik Frisch, Tabea Wilke"
__credits__ = ["Maximilian A. Gehrke", "Yannik P. Frisch", "Tabea A. Wilke"]
__license__ = "GPL"
__version__ = "1.0"
__status__ = "Development"


def find_best_model(env_name):
    """
    This method searches the path "data/" + env_name for the folder which
    contains the model with the best results of the given environment. In this
    case best means the highest average trajectory reward.

    :param env_name: the environment name
    :return: the path to the best model, the average trajectory
    """

    env_path = "data/" + env_name
    best_reward = float("-inf")
    best_model_dir = ""
    best_date = ""

    for root, dirs, files in os.walk("data"):

        if root == env_path:
            for d in dirs:
                results_dir = env_path + "/" + d + "/results.txt"
                try:
                    f = open(results_dir, "r")
                    for line in f:
                        start_with = "Average trajectory reward: "
                        if line.startswith(start_with):
                            reward = float(line[len(start_with):].strip())
                            if reward > best_reward:
                                best_reward = reward
                                best_model_dir = env_path + "/" + d + "/model/"
                                best_date = d

                except FileNotFoundError:
                    pass
            break

    return best_model_dir, best_reward, best_date
