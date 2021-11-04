from __future__ import print_function
import sys
import time
import gym
import quanser_robots
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from challenge1.Regression import Regressor
from challenge1.Utils import *
from challenge1.Evaluation import *

env = gym.make('Pendulum-v2')
reg = Regressor()


def value_iteration(S, A, P, R, gamma, theta):
    print("Value iteration... ")

    # Initialize value function with bad values
    V = np.zeros(shape=np.shape(S)[0] * [np.shape(S)[1] - 1])
    V.fill(-20)
    goal = get_observation_index(env=env, observation_s=S, x=[0, 0])
    V[goal[0]][goal[1]] = 0
    # Fill policy with neutral action (mid of action space)
    PI = np.zeros(shape=np.shape(S)[0] * [np.shape(S)[1] - 1])
    PI.fill(A[0][int(len(A[0][:]) / 2)])
    t = 1
    states = np.stack(np.meshgrid(range(len(S[0][:]) - 1), range(len(S[1][:]) - 1))).T.reshape(-1, 2)
    while True:
        for s0, s1 in states:
            v = V[s0][s1]
            Qsa = np.zeros(shape=np.shape(A)[0] * [np.shape(A)[1] - 1])
            for a in range(len(A[0][:]) - 1):
                """
                next_state = reg.regressorState.predict(np.array([S[0][s0], S[1][s1], A[0][a]]).reshape(1, -1))[0]
                ns = get_index(env=env, space=S, x=next_state)
                ns0 = ns[0]
                ns1 = ns[1]
                Qsa[a] = P[s0][s1][a][ns0][ns1]*(R[ns0][ns1]+gamma*V[ns0][ns1])
                """
                Qsa[a] = sum(map(sum, (P[s0][s1][a][:][:] * (R[:][:] + gamma * V[:][:]))))
            max_Qsa = np.max(Qsa)
            V[s0][s1] = max_Qsa
            delta = np.abs(v - max_Qsa)
        # Reduce discount factor per timestep
        # gamma = gamma/t
        t += 1
        print("Delta =", delta, end='')
        if delta < theta:
            print(" ... done")
            break
        else:
            print(" > Theta =", theta)
    print()

    # Define policy
    print("Defining Policy ...", end='')
    sys.stdout.flush()
    for s0, s1 in states:
        Qsa = np.zeros(shape=np.shape(A)[0] * [np.shape(A)[1] - 1])
        for a in range(len(A[0][:]) - 1):
            """
            next_state = reg.regressorState.predict(np.array([S[0][s0], S[1][s1], A[0][a]]).reshape(1, -1))[0]
            ns = get_observation_index(env=env, observation_s=S, x=next_state)
            ns0 = ns[0]
            ns1 = ns[1]
            Qsa[a] = P[s0][s1][a][ns0][ns1] * (R[ns0][ns1] + gamma * V[ns0][ns1])
            """
            Qsa[a] = sum(map(sum, (P[s0][s1][a][:][:] * (R[:][:] + gamma * V[:][:]))))
        # Get action for argmax index
        max_index = np.argmax(Qsa)
        max_action = A[0][max_index]
        PI[s0][s1] = max_action
    print("done")
    return V, PI


"""
def evaluate_discrete_space(S, A, gaussian_sigmas):

    P = np.zeros(shape=(np.shape(S)[0] * [np.shape(S)[1]-1] +
                        np.shape(A)[0] * [np.shape(A)[1]-1] +
                        np.shape(S)[0] * [np.shape(S)[1]-1]))
    # R = np.zeros(shape=(S_shape[0] * [S_shape[1]] + A_shape[0] * [A_shape[1]]))

    # This reward 'function' is only defined for the successor states
    R = np.zeros(shape=(np.shape(S)[0] * [np.shape(S)[1]-1]))

    # This part is defining the value function by predicting the reward of every discrete
    # state action pair
    # TODO: more efficient way?
    print("Evaluating reward function ... ", end='')
    sys.stdout.flush()
    states = np.stack(np.meshgrid(range(R.shape[0]),range(R.shape[1]))).T.reshape(-1,2)
    for s0, s1 in states:
        R[s0][s1] = reg.regressorReward.predict(np.array([S[0][s0], S[1][s1]]).reshape(1, -1))
    print("done\n")

    # This part is defining the state transition prob. by predicting the resulting state
    # for every state action pair
    # TODO: more efficent way?
    print("Evaluating state transition function ... ", end='')
    sys.stdout.flush()
    states_action = np.stack(np.meshgrid(range(P.shape[0]),range(P.shape[1]), range(P.shape[2]))).T.reshape(-1, 3)
    for s0, s1, a in states_action:
        # Successor of state (s0, s1) for action a
        # We use [0] because we only have one state
        next_state = reg.regressorState.predict(np.array([S[0][s0], S[1][s1], A[0][a]]).reshape(1, -1))[0]
        # Get discrete index of next state
        ns = get_observation_index(env=env, observation_s=S, x=next_state)
        ns0 = ns[0]
        ns1 = ns[1]

        #main_p = 1.0
        #P[s0][s1][a][ns0][ns1] = main_p

        successor_indices = np.stack(np.meshgrid(range(P.shape[3]), range(P.shape[4]))).T.reshape(-1, 2)

        cov = np.eye(2, 2)
        cov[0][0] = gaussian_sigmas[0]
        cov[0][1] = 0
        cov[0][1] = 0
        cov[1][1] = gaussian_sigmas[1]

        max_index0 = np.shape(P)[0]

        for i0, i1 in successor_indices:
            P[s0][s1][a][i0][i1] = 0.5 * (multivariate_normal.pdf(x=np.array([i0, i1]),
                                                           mean=np.array([ns0, ns1]), cov=cov)
                                          + multivariate_normal.pdf(x=np.array([i0, i1]),
                                                            mean=np.array([max_index0+ns0, ns1]), cov=cov))
    print("done\n")

    return P, R
"""


def main(make_plots):
    # Start time
    start = time.time()

    # Define discrete spaces
    S, A = build_discrete_space(env=env)
    # Sample
    # Use save_flag=False to load from sample file
    P, R = reg.sample(env=env, S=S, A=A, gaussian_sigmas=np.array([1, 1]), epochs=10000, save_flag=True)
    # P, R = evaluate_discrete_space(S=S, A=A, gaussian_sigmas=np.array([1, 1]))
    # Do value iteration
    V, PI = value_iteration(S=S, A=A, P=P, R=R, gamma=0.75, theta=1e-15)

    # End time
    end = time.time()
    print("\nTime elapsed: {} seconds \n".format(np.round(end - start, decimals=2)))

    # Evaluate policy from value iteration
    state_distribution = evaluate(env=env, S=S, episodes=10, policy=PI, render=True, sleep=0, epsilon_greedy=0.0)

    if make_plots:
        visualize(value_function=V, policy=PI, R=R, state_distribution=state_distribution, state_space=S)


if __name__ == "__main__":
    main(make_plots=True)
