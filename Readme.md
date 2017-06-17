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
is 44 %.

Architecture
---
The neural network is a four layer network of the following structure:

`IN (N x 3072) -> H1 (1000 x 1) -> H2 (250 x 1) -> H3 (100 x 1) -> OUT (10 x 1)`

where N is the number of input observations (training images).

The size of the input layer `IN` and output layer `OUT` are fixed (as the amount of pixels and colors doesn't change).
The sizes of the hidden layers are just the result of an _educated_ guess, i.e. they might not be optimized for the
problem.

### Input layer
Input layer data is given as a matrix consisting of rows for each input observation (image). Each image has 3 color
channels for each of the 32 x 32 pixels. This data is flattened out into a vector of length 3072 (3 x 32 x 32).

### Hidden layer
Hidden layers are integral to the performance of the network. They are fully connected. Each neuron in the layer
receives all outputs of the previous layer's neurons, weighted by the weight parameter (i.e. connection weights).

Every hidden layer uses ReLU as an activation function. ReLU simply takes the input to the neuron (the dot product of
the inputs and the weights) and passes it on when it's positive or outputs 0 when it's negative.

To prevent overfitting the training set, this network makes use of two strategies:
1. Minibatch gradient descent, whereby the training isn't done over the whole training set (50000 observations). During
training, a random subset of the training set (e.g. a batch of 512 observations) is used. This has the additional
advantage of significantly reducing training time.
2. Dropout, whereby _learned neurons_ are simply dropped with a certain probability.

### Output layer
The output layer contains the class scores for each input observation. There are 10 scores, each representing a class.
The highest class should be the correct class.

The objective of the network is a cross-entropy loss function. This function takes the scores of the output layer and
compares it to the ground truth labels of the data. The loss function yields a high value when the prediction doesn't
match the ground truth labels and a low value if otherwise.

Adam is used to update learnable parameters.

Computational caveats
---
To speed up performance, the network uses numpy matrix operations. Every network layer and learnable parameter can be
represented as matrices and their operations (e.g. dot product). That way, the time needed to train the network
decreases substantially.

Another option would have been to use [minpy](https://minpy.readthedocs.io/en/latest/index.html), which basically
provides a numpy interface but uses the GPU under the hood. Normally, the GPU exceeds CPU performance on tasks like
matrix operations. When tested on this network though, the GPU performed significantly worse. Maybe the numpy code isn't
exactly optimized for GPU. The test network with Keras (using TensorFlow) performed better on the GPU.


Conclusio
---
The accuracy is clearly not enough for real-world scenarios. The problem to date is that due to limited computational
power, the network cannot take all training images into account. A fully connected neural network is very expensive both
in CPU and memory usage. That's why
[Convolutional neural networks](https://en.wikipedia.org/wiki/Convolutional_neural_network) were devised. These are
neural networks designed specifically for the task of image classification. Coding the neural network as convolutional
network would greatly increase performance.