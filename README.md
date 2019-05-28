# Exploring 3D environments using Deep Reinforcement learning techniques



This project contains code for training Deep reinforcement learning algorithm on the First-Person Shooter game Doom.

It was made in the context of the Theory and Methods for Reinforcement Learning lectures given at EPFL by Prof. Volkan Cevher.



The implemented algorithms are:

- Double Deep Q-Network
- Double Deep Recurrent Q-Network
- Deep Q-Network with Variational Auto-Encoder



## environment

This folder contains a wrapper for the ViZDoom environment in openai-gym standards.

The Behaviour class provides a convenient way to shape the reward without manually editing the Doom files.

This package also provides the maps and scenarios used for the project.  

It is meant to be installed with pip. This will also install the requirements necessary to run a ViZDoom environment.



## src

This folder contains the reinforcement learning code used for the experiments. Pytorch must be installed to run the training scripts. The code was written for Python 3.

Its structure is the following:

- Agents: contains the implementation of the RL agent and the replay buffer
- Behaviours: contains the reward shaping and game configurations for ViZDoom
- Datasets: contains the code used to generate, process and load data for training
- Models: contains the neural networks used in the agents
- Plotting: contains the class used to plot the agent behaviour
- Testing: contains the scripts used to test trained agents
- Training: contains the scripts used to train the different models



All the code in this repository is the work of the following authors:

*François Marelli*

*Angel Martínez-González*

*Bastian Schnell*

