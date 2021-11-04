<h2>Episodic Natural Actor Critic (eNAC)</h2>

This project, incorporates the episodic natural actor critic algorithm by Jan Peters and Stefan Schaal (https://doi.org/10.1016/j.neucom.2007.11.026) using Neural Networks and Tensorflow 1.9.

The episodic natural actor critic has several important properties:

- eNAC uses real episodes sampled from the environment and applies an update after a given number of steps.
- eNAC applies the natural gradient which has several advantages over the vanilla gradient ([see: Natural Gradient Works Efficiently in Learning by Shun-ichi Amari](https://doi.org/10.1162/089976698300017746 )), which is, for example, that it is parametrization invariant.
- eNAC uses an actor-critic approach which approximates the value function as well as the policy using Neural Networks in tensorflow.



## Environments

The natural actor critic algorithm at hand makes it very easy to run _Gym_ environments (https://gym.openai.com/) and _Quanser Robots_ environments (https://git.ias.informatik.tu-darmstadt.de/quanser/clients). 

_Gym_ and _Quanser Robots_ environments work out of the box, however, it is easily possible to adapt every environment by using the MyEnvironment class in the _my_environment.py_ file. 



<h2> Files </h2>

`main.py:` 

* This file is the key entry point for using the eNAC algorithm. 
* You can chose between different environments and specify their hyperparameters.
* Additionally you can chose if you want to use pretrained weights and if you want to train, evaluate and/or render and the environemnt.

`my_environment.py:`

* This file contains the _MyEnvironment_ class, which is used to wrap the environments we want to solve.
* All _Gym_ and _Quanser Robots_ environments should be already adapted to the system.
* If you want to run another environment, this is the only file that has to be changed.
* For easy implementation, conditions asking for the name of the environment, can be used.

`nac.py:`

* This class records transitions by executing steps in the environment, which uses the predicted actions of our actor neural network.

*  Afterwards, it calculates the discounted return and the critic's value function prediction for every state and subtracts the latter from the first to get the advantages.
* Finally, it updates the critic by using the discounted returns and the actor by using the advantages.

`actor.py:` Constructs and embodies the actor of our natural actor critic algorithm.

`critic.py:` Constructs and embodies the critic of our natural actor critic algorithm.

`utilities.py:` Some utility functions used for reading and writing data



<h2> Hperparameters </h2>

All Hyperparmeters can be set in the `main.py` file from line 55 to 110. A dictionary (_env_dict_) is used to save different hyper parameter settings, using an integer as key. Before every run of the `main.py` file, the correct set of hyper parameters must be selected by assingning the dictionary key to the variable `ENVIRONMENT`.

The hyper parameter settings must be saved inside the _env_dict_ dictionary in a list which contains 9 elements. In the following, each index and the corresponding hyper parameter has been described:

```
0: Name of the Gym/Quanser environment.
1: If the environment is descrete or continuous.
2: Chose the discretization of continuous environments.
   If the environment is already discrete, put [0].
   you can also exactly specifiy the discretization.
3: Batch size. How much steps should the agent perform before updating 
   parameters. If the trajectory ends before that (done == True), a new 
   trajectory is started.
4: How many updates (of parameters) do we want.
5: Discount factor for expected monte carlo return.
6: Learning rate for actor model. sqrt(learning_rate/ Grad_j^T * F^-1).
7: Learning rate for Adam optimizer in the critic model.
8: The hidden layer size of the critic network.
```



## Starting eNAC

1. Activate the virtual environment with `conda activate group19`.
2. Start the main method with `python3.6 DDPG.py`.



## Training, Evaluating and Rendering (+ Saving)

In the `main.py` file, the following boolean variables exist:

```
LOAD_WEIGHTS = None
TRAIN = True
EVALUATION = True
RENDER = True
```

Each stands for a step in our execution pipeline:

* *LOAD_WEIGHTS*: If a file name is specified, the weights of the specified file will be loaded before training, evaluation and rendering.
* *TRAIN*: The actor and critic network will be trained on our environment. This is compatible with loading weights beforehand.
* *EVALUATION*: The policy (= our actor) is evaluated in the environment. By default, the evaluation is done on 100 samples. After evaluation the results are saved as text and plots in a folder in `./data/<environment>/<current date>`. This is also the place where we **save** the data, so if you want to have something saved, you need to execute the evaluation.
* *RENDER*: Set to True to run episodes (by default 10) in the environment. The action the agent takes are generated by the (pre-)trained actor.