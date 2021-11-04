import gym
import quanser_robots
import numpy as np
from MemoryDQN import MemoryDQN
from DQNNet import DQN
import math
import random
import matplotlib.pyplot as plt
import datetime


import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F


class ActionDisc(gym.Space):
    # self defined action space
    def __init__(self, high, low, number):
        gym.Space.__init__(self, (), np.float)
        self.high = high
        self.low = low
        self.n = number
        self.space = np.linspace(self.low, self.high, self.n)
        #print(self.space)

    def sample(self):
        """
        Random sample from the discrete action space.
        :return: Random action sample
        """
        return random.choice(self.space)

    def contains(self, x):
        """
        :param x: action space values (as array or similar)
        :return: index of the given actions (as list)
        """
        indices = []
        for i in x:
            indices.append(np.where(self.space==i)[0][0])
        return indices


def model_to_action(model, s, env=gym.make("CartpoleSwingShort-v0"), actionspace=10):
    """
    Gives an action for a given state from the model.
    :param model: The pytorch model for the approximated Q(s, a)
    :param s: State of the gym environment.
    :param env: The gym environment.
    :param actionspace: Size of discrete action space
    :return: Action
    """
    env.action_space = ActionDisc(env.action_space.high, env.action_space.low, actionspace)
    pred_action = model(s)
    max_action = pred_action.max(1)[1]
    # return the best action as ndarray
    return torch.tensor(np.array([[env.action_space.space[max_action]]])).numpy()[0]

def run_dqn(env, save = False):
    """
    Runs the DQN algorithm.
    :param env: Gym environment
    :param save: Set True to save pytorch model of learned weights.
    :return: The learned pytorch model
    """
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor

    EPISODES = 1
    #EPISODES = 2000
    BATCH_SIZE =1000
    GAMMA = 0.9
    HIDDEN_LAYER_NEURONS = 300
    LEARNING_RATE = 0.0001
    ACTION_SPACE = 10  # 49
    EPS_START = 1
    EPS_END = 0.01
    #EPS_END = 0.05
    EXPLORATION_STEPS = 1e5

    INITIAL_REPLAY = 100
    REPLAY_SIZE = 1e6
    TARGET_UPDATE = 7000
    global EPSILON
    EPSILON = EPS_START
    EPSILON_STEP = (EPS_START - EPS_END) / EXPLORATION_STEPS

    # define a new discrete action space
    env.action_space = ActionDisc(env.action_space.high, env.action_space.low, ACTION_SPACE)

    # create the replay buffer and the neural networks
    memory = MemoryDQN(REPLAY_SIZE)
    model = DQN(HIDDEN_LAYER_NEURONS, ACTION_SPACE, env.observation_space.shape[0])
    target = DQN(HIDDEN_LAYER_NEURONS, ACTION_SPACE, env.observation_space.shape[0])

    target.load_state_dict(model.state_dict())

    #target.l1.weight = model.l1.weight
    #target.l2.weight = model.l2.weight
    #target.l3.weight = model.l3.weight

    optimizer = optim.Adam(model.parameters(), LEARNING_RATE)

    cum_reward = []

    def select_action(state_pred):
        """
        Epsilon greedy policy
        :param state_pred: Curren state
        :return: Action
        """
        sample = random.random()
        global EPSILON
        epsilon_old = EPSILON
        if EPSILON > EPS_END and memory.size_mem() > INITIAL_REPLAY:
            EPSILON -= EPSILON_STEP
        #epsilon_old = 0.05
        if sample > epsilon_old and memory.size_mem() > INITIAL_REPLAY:
            with torch.no_grad():
                # predict the actions to the given states
                pred_actions = model(state_pred)
                # find the action with the best q-value
                max_action = pred_actions.max(1)[1]
                # return the best action as tensor
                return torch.tensor(np.array([[env.action_space.space[max_action]]]))
        # exploration
        else:
            # return a random action of the action space
            return torch.tensor(np.array([[env.action_space.sample()]]))

    total_steps = 1
    # Start time
    start = datetime.datetime.now()
    for epi in range(EPISODES):
        cum_reward.append(0)
        state = env.reset()
        step = 0
        total_loss = 0

        while True:
            action = select_action(state)

            state_follows, reward, done, info = env.step(action.numpy()[0])

            cum_reward[epi] += reward

            memory.add_observation(state, action, reward, state_follows)

            # if epi == EPISODES - 1:
            #    env.render()


            # training
            if memory.size_mem() > BATCH_SIZE:
            #if memory.size_mem() > INITIAL_REPLAY:

                states, actions, rewards, next_states = \
                    memory.random_batch(BATCH_SIZE)

                # find the index to the given action
                actions = env.action_space.contains(actions)
                # repeat it for the gather method of torch
                actions = np.array(actions).repeat(ACTION_SPACE) \
                    .reshape(BATCH_SIZE, ACTION_SPACE)
                # change it to a long tensor (instead of a float tensor)
                actions = LongTensor(actions)

                # for each q-value(for each state in the batch and for each action)
                # take the one from the chosen action

                current_q_values = model(states)[0].gather(dim=1, index=actions)[:, 0]

                # neural net estimates the q-values for the next states
                # take the ones with the highest values
                #max_next_q_values = model(next_states)[0].detach().max(1)[0]
                max_next_q_values = target(next_states)[0].detach().max(1)[0]

                expected_q_values = rewards + (GAMMA * max_next_q_values)

                #loss = F.smooth_l1_loss(current_q_values, expected_q_values.type(FloatTensor))
                loss = F.mse_loss(current_q_values, expected_q_values.type(FloatTensor))
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_steps += 1
                # update the model weights with the target parameters
                if total_steps % TARGET_UPDATE == 0:
                    total_steps = 1
                    target.load_state_dict(model.state_dict())
                    #target.l1.weight = model.l1.weight
                    #target.l2.weight = model.l2.weight
                    # target.l1.weight = 0.001*model.l1.weight+(1-0.001)*target.l1.weight
                    # target.l2.weight = 0.001*model.l2.weight+(1-0.001)*target.l2.weight
                    #target.l3.weight = model.l3.weight

            state = state_follows
            step += 1

            if done:
                cum_reward[-1]=cum_reward[-1]/step
                break
            '''if step == 500:
                cum_reward[-1]=cum_reward[-1]/500.
                break'''
        # print("Episode:{} Steps:{} Cum.Reward:{} Loss/Step:{} Epsilon:{}"
        #      .format(epi, step, cum_reward[-1], total_loss/step, EPSILON))
    # End time
    end = datetime.datetime.now()
    # print("Learning took", (end-start))
    if save:
        torch.save(model, "model.pt")
    # plt.plot(cum_reward)
    # plt.show()
    return model


env = gym.make("CartpoleSwingShort-v0")

run_dqn(env, save=False)
# run_dqn(env, save=True)

