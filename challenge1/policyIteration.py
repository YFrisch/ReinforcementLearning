from __future__ import print_function
import sys
import numpy as np
import gym
import quanser_robots
from Regression import Regressor
from Evaluation import *
from Utils import *


env = gym.make('Pendulum-v2')
reg = Regressor()


def __policy_evaluation(S, A, P, R, PI, V, theta, gamma):
    print("Policy Evaluation ... ")
    while True:
        delta = 0
        states = np.stack(np.meshgrid(range(len(S[0][:]) - 1), range(len(S[1][:]) - 1))).T.reshape(-1, 2)
        for s0, s1 in states:
            v = V[s0][s1]
            a = PI[s0][s1]
            # get index of action
            a = get_action_index(env=env, action_s=A, a=np.array([a]))
            expected_reward = sum(map(sum, (P[s0][s1][a][0][:][:] * (R[:][:] + gamma * V[:][:]))))
            V[s0][s1] = expected_reward
            delta = max(delta, np.abs(v-expected_reward))
        print("Delta =", delta, end='')
        if delta < theta:
            print(" ... done!")
            break
        else:
            print(" > Theta =", theta)
    return V


def __policy_improvement(S, A, P, R, PI, V, gamma):
    print("Policy Improvement ... ",end='')
    sys.stdout.flush()
    states = np.stack(np.meshgrid(range(len(S[0][:])-1),range(len(S[1][:])-1))).T.reshape(-1, 2)
    policy_stable = True
    for s0, s1 in states:
        b = PI[s0][s1]
        Qsa = np.zeros(shape=np.shape(A)[0] * [np.shape(A)[1] - 1])
        for a in range(len(A[0][:]) - 1):
            Qsa[a] = sum(map(sum, (P[s0][s1][a][0][:][:] * (R[:][:] + gamma * V[:][:]))))
        # Get action for argmax index
        max_index = np.argmax(Qsa)
        max_action = A[0][max_index]
        PI[s0][s1] = max_action
        if b != max_action:
            policy_stable = False
    if policy_stable:
        print("policy stable!")
    else:
        print("policy NOT stable!")
    return PI, policy_stable


def policy_iteration(S, A, P, R, theta, gamma):

    print("Policy Iteration ... ")

    # Initialize value function and policy
    V = np.zeros(shape=np.shape(S)[0] * [np.shape(S)[1] - 1])
    PI = np.zeros(shape=np.shape(S)[0] * [np.shape(S)[1] - 1])

    policy_stable = False
    while not policy_stable:
        V = __policy_evaluation(S, A, P, R, PI, V, theta, gamma)
        PI, policy_stable = __policy_improvement(S, A, P, R, PI, V, gamma)

    print("done!")

    return V, PI

def main(make_plots):                     action_space_size=(16+1,))
    # Start time
    start = time.time()
	
	# Build discrete spaces
    S, A = build_discrete_space(env=env)
	# Sample and estimate reward and transition function
	# Use save_flag=False to load a saved sample file
    P, R = reg.sample(env=env, S=S, A=A, gaussian_sigmas=np.array([1, 1]), epochs=10000, save_flag=True)
	# Do policy iteration
    V, PI = policy_iteration(S=S, A=A, P=P, R=R, gamma=0.95, theta=1e-15)

    # End time
    end = time.time()
    print("\nTime elapsed: {} seconds \n".format(np.round(end - start, decimals=2)))

	# Evaluate policy from policy iteration
    state_distribution = evaluate(env=env, S=S, episodes=10, policy=PI, render=False, sleep=0, epsilon_greedy=0.0)

    if make_plots:
        visualize(value_function=V, policy=PI, R=R, state_distribution=state_distribution, state_space=S)


if __name__ == "__main__":
    main(make_plots=True)
