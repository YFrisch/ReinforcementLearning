import gym
import time
import numpy as np
import TrueModel


env = gym.make('Pendulum-v2') # Create gym/quanser environment
env.reset()

counter = 20
positive = True

for i in range(200):
    print("-----------------------")
    env.render()
    time.sleep(.5)
    if positive:
        a = 2.0
    else:
        a = -2.0
    state, reward, done, info = env.step(np.array([a]))
    print("Action: ", a)
    print("State: ", state)
    print("Reward: ", reward)
    print("Done:  ", done)

    counter -= 1
    if counter == 0:
        positive = not positive
        counter = 5