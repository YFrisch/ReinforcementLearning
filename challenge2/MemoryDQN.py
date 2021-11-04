import random
import numpy as np
from collections import deque
import torch
from torch.autograd import Variable


class MemoryDQN(object):
    def __init__(self, buffer_number):
        self.size = buffer_number
        self.ReplayBuffer = deque()

    def add_observation(self, state, action, reward, next_state):
        """
        Adds observation to the replay buffer.
        :param state: State of the gym env
        :param action: Action of gym env
        :param reward: Observed reward
        :param next_state: Observed next state
        :return: None
        """
        self.ReplayBuffer.append([state, action, reward, next_state])
        if len(self.ReplayBuffer) > self.size:
            self.ReplayBuffer.popleft()

    def random_batch(self, batch_size):
        """
        Samples a random mini-batch from the replay buffer
        :param batch_size: mini-batch size of random sample
        :return: The mini-batch of experiences
        """
        element_number = len(self.ReplayBuffer)
        if element_number < batch_size:
            batch_size = element_number
        expectations = random.sample(self.ReplayBuffer, k=batch_size)
        states = list(zip(*expectations))[0]
        actions = list(zip(*expectations))[1]
        rewards = list(zip(*expectations))[2]
        next_states = list(zip(*expectations))[3]
        return np.array(states), np.array(actions), \
               torch.from_numpy(np.array(rewards)).type(torch.FloatTensor),np.array(next_states)

    def size_mem(self):
        """
        Returns current size of replay buffer.
        :return: Size of replay buffer memory
        """
        return len(self.ReplayBuffer)
