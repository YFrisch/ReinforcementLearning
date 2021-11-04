from __future__ import print_function
import sys
import matplotlib.pyplot as plt
import numpy as np
import time
from Utils import *

"""
    Evaluation stuff to see the predictions, discretizations and learned functions in action
"""

def evaluate(env, S, episodes, policy, render, sleep, epsilon_greedy=None):

    state_distribution = np.zeros(shape=np.shape(S)[0]*[np.shape(S)[1]-1])

    rewards_per_episode = []
    print("Evaluating...")
    sys.stdout.flush()

    for e in range(episodes):

        state = env.reset()

        cumulative_reward = [0]

        for t in range(200):
            # Render environment
            if render:
                env.render()
                time.sleep(sleep)

            # discretize state
            index = get_observation_index(env, S, state)
            state_distribution[index[0]][index[1]] += 1

            if epsilon_greedy is not None:
                rand = np.random.rand()
                if rand < epsilon_greedy:
                    action = np.random.uniform(low=env.action_space.low, high=env.action_space.high)
                else:
                    action = np.array([policy[index[0], index[1]]])
            else:
                # Do step according to policy and get observation and reward
                action = np.array([policy[index[0], index[1]]])

            next_state, reward, done, info = env.step(action)
            state = np.copy(next_state)

            cumulative_reward.append(cumulative_reward[-1] + reward)

            if done:
                print("Episode {} finished after {} timesteps".format(e + 1, t + 1))
                break

        rewards_per_episode.append(cumulative_reward)

    print("...done")

    # Average reward over episodes
    rewards = np.average(rewards_per_episode, axis=0)

    env.close()

    # Plot rewards per timestep averaged over episodes
    plt.figure()
    plt.plot(rewards, label='Cumulative reward per timestep, averaged over {} episodes'.format(episodes))
    plt.legend()
    plt.show()

    return state_distribution
