import numpy as np
import gym
import matplotlib.pyplot as plt
import quanser_robots

# ---------
# PARAMETERS
EPISODES = 1000000
STATE_SPACE_SIZE = (30 + 1, 30 + 1)
ACTION_SPACE_SIZE = (30 + 1)

# ---------
env = gym.make("Pendulum-v2")

action_space = (np.linspace(env.action_space.low, env.action_space.high,
                            ACTION_SPACE_SIZE))
state_space = (np.linspace(-np.pi, np.pi, STATE_SPACE_SIZE[0]),
               np.linspace(-8, 8, STATE_SPACE_SIZE[1]))
# ---------


def discretize_index(state):

    radian_diff =  [np.abs(x - state[0]) for x in state_space[0]]
    vel_diff = [np.abs(x - state[1]) for x in state_space[1]]

    index = np.array([np.argmin(radian_diff), np.argmin(vel_diff)])

    return index

# ---------


state = env.reset()
distribution = np.zeros(STATE_SPACE_SIZE)

for e in range(EPISODES):
    a = env.action_space.sample()
    s, r, dones, infos = env.step(a)
    index = discretize_index(s)
    distribution[index[0], index[1]] += 1

    if(e % 1000 == 0):
        print("{} episodes".format(e))


plt.title("State Distribution")
plt.imshow(distribution)
plt.colorbar()

if state_space is not None:
    plt.ylabel("Angle in Radians")
    plt.yticks(range(len(state_space[0])), labels=state_space[0].round(2))
    plt.xlabel("Velocity")
    plt.xticks(range(len(state_space[1])), labels=state_space[1].round(1))

plt.show()