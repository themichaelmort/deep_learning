This repository contains the code for several deep learning projects. All projects are implemented 
in PyTorch and presuppose that the user has a cuda-enabled GPU.

(c) Michael Mortenson

Projects
--------

image_classification.py - Simple classification
    This project implements a custom vanilla fully-connected neural network with two 1000-neuron 
    layers with ReLu as the activation function. It makes use of the publically available 
    FashionMNIST dataset from torchvision. It defaults to running 100 epochs of training with a 
    batch size of 42 using Stochastic Gradient Descent for the optimizer. PyTorch's 
    CrossEntropyLoss is used for the loss function. Work remains to get a high accuracy with this 
    model, but it exists as an example of classification using a neural network.
