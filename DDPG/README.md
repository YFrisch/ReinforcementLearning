<h2> Deep Deterministic Policy Gradient </h2>

<ul><li>DDPG.py: The main file for DDPG with the complete algorithm </li>
<li>NeuralNetworks.py: The file which creates the actor and the critic network</li>
<li>ReplayBuffer.py: Defines the memory of the algorithm with its own datastructure</li>
<li>ActionNoise.py: Noise which is added to the chosen action</li></ul>

<h4> Brief introduction to DDPG</h4>
DDPG is an reinforcement learning algorithm which combines the approaches of DPG and DQN. It is a model-free and off-policy algorithm and uses actor-critic methods with a deterministic target policy and deep Q-Learning. The critic is updated with the Bellman Equation with TD-error and the actor is updated using the DPG theorem. The replay buffer is used to sample random mini-batches to ensure independently and identically distributed data which are decorrelated. The target network is constrained to slow changes what greatly improves stability.

<h4>Starting DDPG</h4>
1. Please activate the virtual environment via <br/>
    <i>conda activate group19</i><br/>
2. Start the main method with <br/>
<i> python3.6 DDPG.py</i>

<h4>Hyperparameter Changes </h4>
1. To change the deep neural network structures, please edit NeuralNetworks.py. Setting num_layers=1 in the objects 
header automaticall uses one hidden layer with fc1_units hidden units. <br/>
2. Replay Buffer specifications can be directly changed in line #326 in main() method of DDPG.py. E.g. one could change 
the batch_size here to 1024 or back to 64, aswell as the total buffer size. <br/>
3. The action noise object has to be initialized in main() method of DDPG.py. Hyperparameters can also be set there. 
E.g. the standard deviation sigma of the gaussion noise might be set to 0.1 in line #323. To implement different 
noises, pleasy directly modifiy AcitonNoise.py <br/>
4. The remaining hyperparameters are defined in the call of the training() method in line #330 in the main() method.
 One could run the algorithm with less discounting by changing to gamma=x in line #332 <br/>

<h4>Saving and Loading </h4>
1. The algorithm automatically saves the actor and critic pytorch models and the training progress plots if save_flag is
set to 'True' in line #349 in the main() method in DDPG.py. They are saved in a folder formated by (day-month-hour).
2. Performance plots for evaluation are shown automatically and can be saved by hand. To render the evaluation episodes,
set evaluate to 'True ' in line #354 in main() method in DDPG.py
3. To use a pretrained actor for evaluation, please set load_flag to 'True' and define the load_path (day-month-hour)
, containing the actor model, in line #350 in DDPG.py
4. To use a pretrained actor and critic model to continue training, please set use_pretrained to 'True' and define the 
load_path