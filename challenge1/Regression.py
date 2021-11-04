import sys
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

from Utils import *


class Regressor:

    def __init__(self):
        self.regressorReward = RandomForestRegressor(n_estimators=10, min_samples_split=2)
        self.regressorState = RandomForestRegressor(n_estimators=20, min_samples_split=2) # currently not used
		# Transition probabilites
        self.P = None
		# Reward function
        self.R = None

    # Performs regression from given state and performed action
    # to successive state and observed reward
    def sample(self, env, S, A, gaussian_sigmas, epochs, save_flag):

        if not save_flag and open('./pickle/sample.pkl'):
            print()
            print("Found sample file.")
            print()
            with open('./pickle/sample.pkl', 'rb') as pickle_file:
                (rS, rR, P, R) = pickle.load(pickle_file)
            self.regressorState = rS
            self.regressorReward = rR
            self.R = R
            self.P = P

        else:
            rtx = []
            rty = []
            stx = []
            sty = []
            plotr = []
            plots = []
            old_state = env.reset()

            self.P = np.zeros(shape=(np.shape(S)[0] * [np.shape(S)[1] - 1] +
                                np.shape(A)[0] * [np.shape(A)[1] - 1] +
                                np.shape(S)[0] * [np.shape(S)[1] - 1]))

            # This reward 'function' is only defined for the successor states
            self.R = np.zeros(shape=(np.shape(S)[0] * [np.shape(S)[1] - 1]))
            self.R.fill(-20)

            # Distribution of samples
            self.sample_distribution = np.zeros(shape=(np.shape(S)[0] * [np.shape(S)[1] - 1]))

            print("Sampling: 0% ... ", end='')
            sys.stdout.flush()

            for i in range(epochs):
                action = env.action_space.sample()
                next_state, reward, done, info = env.step(action)

                #old_state[0] += 2*np.pi
                #next_state[0] += 2*np.pi

                #rtx.append(np.append(old_state, action))
                rtx.append(next_state)
                rty.append(reward)
                stx.append(np.append(old_state, action))
                sty.append(next_state)

                # Update state transition probabilites
                state_index = get_observation_index(env=env, observation_s=S, x=np.array([old_state[0], old_state[1]]))
                s0, s1 = state_index[0], state_index[1]
                self.sample_distribution[s0][s1] +=1
                action_index = get_action_index(env=env, action_s=A, a=np.array([action]))
                a = action_index[0]
                next_state_index = get_observation_index(env=env, observation_s=S,
                                                         x=np.array([next_state[0], next_state[1]]))
                ns0, ns1 = next_state_index[0], next_state_index[1]
                self.P[s0][s1][a][ns0][ns1] += 1

                if i % 50 == 0:  # 50 works nicely

                    # Regression from next observed state to observed reward
                    self.regressorReward.fit(rtx, rty)
                    fitrtx = self.regressorReward.predict(rtx)
                    mse = mean_squared_error(rty, fitrtx)
                    plotr.append(mse)

                    # Regression from state and action to next state
                    self.regressorState.fit(stx, sty)
                    fitstx = self.regressorState.predict(stx)
                    mse = mean_squared_error(sty, fitstx)

                    plots.append(mse)

                old_state = np.copy(next_state)

                if i == int(epochs * 0.25):
                    print("25% ... ", end='')
                    sys.stdout.flush()
                if i == int(epochs * 0.5):
                    print("50% ... ", end='')
                    sys.stdout.flush()
                if i == int(epochs * 0.75):
                    print("75% ... ", end='')

                    sys.stdout.flush()
            print("done!")

            # Normalize transition probabilites
            print("Normalizing transition probabilities ... ",end='')
            sys.stdout.flush()
            states_action = np.stack(np.meshgrid(range(self.P.shape[0]),
                                                 range(self.P.shape[1]), range(self.P.shape[2]))).T.reshape(-1, 3)
            for s0, s1, a in states_action:
                amount = sum(map(sum, self.P[s0][s1][a][:][:]))
                if amount > 0:
                    factor = 1.0 / amount
                    self.P[s0][s1][a][:][:] = np.multiply(self.P[s0][s1][a][:][:], factor)
                else:
                    self.P[s0][s1][a][:][:] = 0

            print("done!")


            # This part is defining the value function by predicting the reward of every discrete
            # state action pair
            print("Evaluating discrete reward function ... ",end='')
            sys.stdout.flush()
            states = np.stack(np.meshgrid(range(self.R.shape[0]), range(self.R.shape[1]))).T.reshape(-1, 2)
            for s0, s1 in states:
                self.R[s0][s1] = self.regressorReward.predict(np.array([S[0][s0], S[1][s1]]).reshape(1, -1))
            print("done!")

            # Plot loss curves
            plt.figure()
            plt.plot(plotr, label="Loss for reward fitting")
            plt.plot(plots, label="Loss for state fitting")
            plt.legend()

            # Plot discrete reward function
            plt.figure()
            plt.imshow(self.R)
            plt.title("Discrete reward function")
            plt.colorbar()
            plt.ylabel("Angle in Radians")
            plt.yticks(range(len(S[0])), labels=S[0].round(2))
            plt.xlabel("Velocity")
            plt.xticks(range(len(S[1])), labels=S[1].round(1))

            # Plot example for state transition function
            plt.figure()
            plt.imshow(self.P[1][1][1][:][:])
            plt.title("State transition probability for S:1|1 A:1")
            plt.colorbar()
            plt.grid()

            # Plot sample state distribution
            plt.figure()
            plt.imshow(self.sample_distribution)
            plt.title("Sample distribution")
            plt.colorbar()
            plt.grid()
            plt.show()

            print("Saving sample file.")
            print()
            save_object((self.regressorState, self.regressorReward, self.P, self.R), './pickle/sample.pkl')

        return self.P, self.R
