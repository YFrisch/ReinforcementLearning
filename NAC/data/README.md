## Data directory

This directory (NAC/data) contains all the data created during training, evaluating and rendering gym or quanser environments.

The saving path follows the same structure for all environments. At top-level, the folder specifies the environment. Each environment folder contains folders and folders only, which have a date as name. The date correspont with the date of execution. Inside these folders is the evaluation data and a file of the used hyperparameters. Additionally, one can find the tensorflow model in the „model“ folder and some data and plots of the training in the „training“ folder.

For more information about the algorithm and the environments, please visit the README.md in the parent directory.