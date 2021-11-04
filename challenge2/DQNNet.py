import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, hidden_neurons, output, input):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(input, hidden_neurons)
        #self.l3 = nn.Linear(hidden_neurons, 50)
        self.l2 = nn.Linear(hidden_neurons, output)

    def forward(self, x):
        """
        Forward pass through the environment.
        :param x: input (state)
        :return: output (approximated V(s))
        """
        x = torch.from_numpy(x).type(torch.FloatTensor).unsqueeze(0)
        x = torch.autograd.Variable(x)
        x = F.relu(self.l1(x))
        #x = F.relu(self.l3(x))
        x = self.l2(x)
        return x
