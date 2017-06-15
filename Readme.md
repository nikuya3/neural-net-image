Neural net for image classification
===
This repository contains a neural network for image classification. The network is coded in Python 3 and depends on
[numpy](http://www.numpy.org/).

Dataset
---
The network uses the well-known [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset. It contains 60000
images which have a size of 32 x 32 pixels. Each image is labelled as one of 10 classes. There are 6000 images per
class. Out of 60000 images, 50000 are used for training and 10000 are used for testing.

Results
---
The objective is to predict the classes of the test dataset (10000 images) based on trained parameters using the 50000
training images. Until now, the best accuracy (defined as number of correct predictions / number of total predictions)
is 16 %.

Architecture
---
The neural network is a three layer network of the following structure:

`IN (N x 3072) -> H1 (1000 x 1) -> H2 (100 x 1) -> OUT (10 x 1)`

where N is the number of input observations (training images).

The size of the input layer `IN` and output layer `OUT` are fixed (as the amount of pixels and colors doesn't change).
The sizes of the two hidden layers are just the result of an _educated_ guess, i.e. they might not be optimized for the
problem.

### Input layer
Input layer data is given as a matrix consisting of rows for each input observation (image). Each image has 3 color
channels for each of the 32 x 32 pixels. This data is flattened out into a vector of length 3072 (3 x 32 x 32).

### Hidden layer
Hidden layers are integral to the performance of the network. They are fully connected. Each neuron in the layer
receives all output of the previous layer's neurons, weighted by the weight parameter (i.e. connection weights).

Every hidden layer uses ReLU as an activation function. ReLU simply takes the input to the neuron (the dot product of the
inputs and the weights) and passes it on when it's positive or outputs 0 when it's negative.

A cross-entropy loss function rates the network performance at test time.

### Output layer
The output layer contains the class scores for each input observation. There are 10 scores, each representing a class.
The highest class should be the correct class.


Conclusio
---
The accuracy is clearly not enough for real-world scenarios. The problem to date is that due to limited computational
power, the network cannot take all training images into account. A fully connected neural network is very expensive both
in CPU and memory usage. That's why
[Convolutional neural networks](https://en.wikipedia.org/wiki/Convolutional_neural_network) were devised. These are
neural networks designed specifically for the task of image classification. Coding the neural network as convolutional
network would greatly increase performance.