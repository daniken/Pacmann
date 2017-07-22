# Reinforcement Learning with Neural Network

In this project we build and train an artificial neural network that plays as pacman. Q-learning was used as the reinforcement learning method. We assume that the environment is spatially isotropic and develop a feature based network. This assumption enables us to build a network that evaluates actions in any direction exactly the same way. The weights of the network are initialized randomly but it is successful in learning how to play well after just a few games with no prior knowledge of how to play. 

The network takes 7 inputs, 2 binary and 5 discrete inputs within the range [1 0). The network currently consists of 10 hidden units and outputs a single-valued number corresponding to the approximate Q-value. The inputs are the following:

* 1 / distance_to_ghost
* 1 / distance_to_pill
* 1 / distance__to_intersection
* 1 / distance_to_capsule
* pills collected in %
* 1 if you're moving to the globally closest pill, 0 otherwise
* 1 if locally closest ghost is scared. 0 otherwise

Due to the isotropic assumption of the environment and the structure of the network, these inputs correspond to only one legal action in the current state. Therefore, for each legal action that we consider in every state we find ourselves in, we extract these features and choose the action that corresponds to the highest network output. 

The feature extraction is done by performing a custom Breadth First Search (BFS) for every objective we want to find. We say custom because we add the initial position to the closed set before we start the search. The search starts at the new position we end up at after performing the action we are extracting features for. Then, if pacman finds himself in a corridor, it will then only start searching in the direction it is considering (only exception is when standing still, where it will do a normal BFS in all directions).

The training is done online and in batches for each turn meaning that we don't only update with respect to the move pacman just made, but have a set of previous experiences that contain the latest transitions done by the agent. The training set is chosen by randomly sampling from the all previous experiences and the network is trained using stochastic gradient decent.

The used environment was developed by John DeNero, Dan Klein, Pieter Abbeel. Visit their [homepage](http://ai.berkeley.edu/project_overview.html) for more info about running the project 

This project is written in python 2.7. The network is build with Keras, having Theano as backend, on a Ubuntu machine. You will also need HDF5 and h5py for saving/loading the network and its weights (read Keras documentation for more info). The main code can be found in qlearningAgents.py  (few lines in learningAgents.py too).
