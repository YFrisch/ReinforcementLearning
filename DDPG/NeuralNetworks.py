import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    """
    Initialize hidden weights according to DDPG paper.
    :param layer: layer of nn
    :return:
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor model, approximating the discrete policy π(s)->a"""

    def __init__(self, state_size, action_size, seed, num_layers=2, fc1_units=40, fc2_units=30):
        """
        Initializes the network's parameters and build it's model.
        :param state_size: The dimension of a state of the environment
        :param action_size: The dimension of an action of the environment
        :param seed: random seed
        :param fc1_units: amount of nodes of first hidden layer  # 400
        :param fc2_units: amount of nodes of second hidden layer  # 300
        """
        super(Actor, self).__init__()
        self.num_layers = num_layers
        self.seed = torch.manual_seed(seed)
        if num_layers == 2:
            self.fc1 = nn.Linear(state_size, fc1_units)
            self.fc2 = nn.Linear(fc1_units, fc2_units)
            self.fc3 = nn.Linear(fc2_units, action_size)
        if num_layers == 1:
            self.fc1 = nn.Linear(state_size, fc1_units)
            self.fc2 = nn.Linear(fc1_units, action_size)
        self.initialize_parameters()

    def initialize_parameters(self):
        """
        Resetting the network's parameters (weights).
        :return: None
        """
        if self.num_layers == 2:
            self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
            self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        if self.num_layers == 1:
            self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
            self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Forward pass through the actor network.
        Mapping states to actions.
        :param state: State input coming from the environment.
        :return: Action from approximate policy from actor network
        """
        if self.num_layers == 2:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            return F.tanh(self.fc3(x))
        if self.num_layers == 1:
            x = F.relu(self.fc1(state))
            return F.tanh(self.fc2(x))


class Critic(nn.Module):
    """Critic model, approximating the value function Q(s,a)."""

    def __init__(self, state_size, action_size, seed, num_layers=2, fcs1_units=40, fc2_units=30):
        """
        Initializes the critic network's parameters and builds it's model.
        The model merges state input and action input after the first hidden layer.
        :param state_size: Dimension of a state of the environment.
        :param action_size: Dimension of an action of the environment.
        :param seed: random seed
        :param fcs1_units: Amount of nodes of first hidden layer  # 400
        :param fc2_units: Amount of nodes of second hidden layer  # 300
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.num_layers = num_layers
        if num_layers == 2:
            self.fcs1 = nn.Linear(state_size, fcs1_units)
            self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
            self.fc3 = nn.Linear(fc2_units, 1)
        if num_layers == 1:
            self.fc1 = nn.Linear(state_size + action_size, fcs1_units)
            self.fc2 = nn.Linear(fcs1_units, 1)
        self.initialize_parameters()

    def initialize_parameters(self):
        """
        Resetting the network's parameters (weights).
        :return: None
        """
        if self.num_layers == 2:
            self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
            self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        if self.num_layers == 1:
            self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
            self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Forward pass through the critic network.
        Mapping state action pairs to q-values.
        :param state: State input coming from the environment
        :param action: Action input coming from the policy / actor network
        :return: Approximate Value function Q(s,a)
        """
        if self.num_layers == 2:
            xs = F.relu(self.fcs1(state))
            x = torch.cat((xs, action), dim=1)
            x = F.relu(self.fc2(x))
            return self.fc3(x)
        if self.num_layers == 1:
            x = F.relu(self.fc1(torch.cat((state, action), dim=1)))
            return self.fc2(x)

