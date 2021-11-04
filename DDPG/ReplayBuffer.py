import numpy as np
import random
from collections import namedtuple, deque
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples and sample random mini-batches from it."""

    def __init__(self, env, buffer_size, batch_size, seed):
        """
        Initializes ReplayBuffer object.
        :param env: The gym environment
        :param buffer_size: Size of the replay buffer; Maximal amount of stored samples
        :param batch_size: Size of the sampled mini-batches
        :param seed:
        """
        self.action_size = len(env.action_space.sample())
        self.env = env
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """
        Adds experience (s, a, r, s', done) to replay buffer.
        :param state: current state of observation
        :param action: current action, chosen by the actor network
        :param reward: observed reward after performing action
        :param next_state: observed next state after performing action
        :param done: observed 'done' flag, inducing finished episodes
        :return: None
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """
        Samples random mini-batch of size 'batch_size' from the replay buffer.
        :return: Mini-batch of randomly sampled experiences
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state.cpu() for e in experiences if e is not None])).float().\
            to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])). \
            float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)). \
            float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """
        Returns the current size of the replay buffer.
        :return: The amound of stored experiences in the replay buffer.
        """
        return len(self.memory)
