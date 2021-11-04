import pickle
import numpy as np
import matplotlib.pyplot as plt

def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def print_array(arr):
    for i in arr:
        print(i)


def build_discrete_space(env):

    """
        Creates discrete observation and action space
    :return:
    """

    state = env.reset()
    observation_size = len(state)
    observation_range = (env.observation_space.low, env.observation_space.high)
    # TODO: Different bins for different dimensions
    observation_bins = observation_size*[20]
    S = []
    for i in range(observation_size):
        S.append(np.linspace(observation_range[0][i], observation_range[1][i], observation_bins[i]))
    #print("Discrete Observation Space: ", S)

    action = env.action_space.sample()
    action_size = len(action)
    action_range = (env.action_space.low, env.action_space.high)
    action_bins = action_size*[10]
    A = []
    for i in range(action_size):
        A.append(np.linspace(action_range[0][i], action_range[1][i], action_bins[i]))
    #print("Discrete Action Space: ", A)

    return S, A


# Returns discrete index of x in space
def get_observation_index(env, observation_s, x):
    index = []
    for dim in range(len(x)):
        for ind in range(len(observation_s[dim][:]) - 1):
            if observation_s[dim][ind] <= x[dim] < observation_s[dim][ind + 1]:
                index.append(ind)
                break
            elif x[dim] == env.observation_space.high[dim]:
                index.append(len(observation_s[dim][:]) - 2)
    return np.array(index)

def get_action_index(env, action_s, a):
    index = []
    for dim in range(len(a)):
        for ind in range(len(action_s[dim][:]) - 1):
            if action_s[dim][ind] <= a[dim] < action_s[dim][ind + 1]:
                index.append(ind)
                break
            elif a[dim] == env.action_space.high[dim]:
                index.append(len(action_s[dim][:]) - 2)
    return np.array(index)


def visualize(value_function, policy, R, state_distribution, state_space=None):
    plt.figure()
    plt.title("Value function")
    plt.imshow(value_function)
    plt.colorbar()
    if state_space is not None:
        plt.ylabel("Angle in Radians")
        plt.yticks(range(len(state_space[0])), labels=state_space[0].round(2))
        plt.xlabel("Velocity")
        plt.xticks(range(len(state_space[1])), labels=state_space[1].round(1))

    plt.figure()
    plt.title("Policy")
    plt.imshow(policy)
    plt.colorbar()
    if state_space is not None:
        plt.ylabel("Angle in Radians")
        plt.yticks(range(len(state_space[0])), labels=state_space[0].round(2))
        plt.xlabel("Velocity")
        plt.xticks(range(len(state_space[1])), labels=state_space[1].round(1))

    plt.figure()
    plt.imshow(state_distribution)
    plt.title("State distribution after evaluating")
    if state_space is not None:
        plt.ylabel("Angle in Radians")
        plt.yticks(range(len(state_space[0])), labels=state_space[0].round(2))
        plt.xlabel("Velocity")
        plt.xticks(range(len(state_space[1])), labels=state_space[1].round(1))

    plt.show()
