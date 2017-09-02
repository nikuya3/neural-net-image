Neural net for image classification
===
This repository contains a multi-layer perceptron (vanilla neural network) implementation in pure Python 3 and depends
on [numpy](http://www.numpy.org/). The network is tested both on the low-dimensional
[Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) and the high-dimensional
[CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

The network performs well on the Iris classification task and reaches a test accuracy of 100 %. On the image
classification task, it reaches a test accuracy of ~40-50 %. This indicates, that this type of neural network does not
scale well to high-dimensional tasks.

Usage
---
There are two ways to try out the neural network.

### test.py
Run `test-iris.py` to train a neural network for Iris classification. It will save the learned parameters to `dump-iris.p`
and print the test accuracy, while also showing sample prediction.

Run `test.py` to train the neural network for image classification. `test.py` will also train an equivalent
Keras network and compare those two networks using their test accuracy metric.

It's also possible to tweak hyperparameters to see how the networks change.

To see the training performance over time, use `plot.py` after running one of the test scripts.

### app.py
To see the image classification neural network _in action_ (i.e. how it would be used in a real use case), run app.py.
This script will apply the learned network by _recognizing objects in an image_ (classifying an image). To do this, run
`./app.py [filename]`. The script takes all images as input, but only detects the objects it has learned (CIFAR-10
classes). You can try the script with the images located in the `img` directory. Of course, only a few predictions 
will be correct, as the network performance is meager.

It cannot be used as a standalone script, as it depends on `parameters.p` and [SciPy](https://www.scipy.org/).

Iris classification
---
The Iris dataset is on of the best known machine learning datasets. It contains 5 values: three independent
real-valued attributes and one dependent factor label. Each observation is classified using one of three classes.
In total, there are 150 observations and the classes are distributed equally.

120 observations (ie. 80 % of the total dataset) are used to train the network, while the remaining 30 observations are
used to calculate the test accuracy. The observations are sampled from a random distribution.

Results
---
The objective is to predict the classes of the test dataset based on trained parameters using the training dataset.
The current test accuracy (defined as `number of correct predictions / number of total predictions`) is 100 %.

### Architecture
The neural network uses a three layer architecture of the following form:

`IN (n x 4) -> H1 (1000 x 1) -> H2 (100 x 1) -> OUT (3 x 1)`

where `n` is the number of input observations (training or test observations).

The size of the input layer `IN` and output layer `OUT` are fixed (as the amount of independent attributes doesn't change).
The sizes of the hidden layers resulted from empirical observation and are not guaranteed to be fully optimized for the
task.

#### Input layer
Input layer data is given as a vector consisting of rows for each observation. As each observation has 4 independent
attributes, the vector is 4-dimensional.

#### Hidden layers
Hidden layers are integral to the performance of the network. They are fully connected. Each neuron in the layer
receives all outputs of the previous layer's neurons, weighted by the weight parameter (i.e. connection weights).

Every hidden layer uses ReLU as an activation function. ReLU simply takes the input to the neuron (the dot product of
the inputs and the weights) and passes it on when it's positive or outputs 0 when it's negative.

To prevent overfitting the training set, this network leverages the
[Dropout technique](http://jmlr.org/papers/v15/srivastava14a.html). Dropout simply drops neurons and their learned
connections with a certain probability. In this case, the probability is 90 %.

#### Output layer
The output layer contains the class scores for each input observation. There are 3 scores, each representing a class.
As the class is a factor label, it is encoded to a number. The highest score should be at the index corresponding the
class number.

The objective of the network is a cross-entropy loss function. This function takes the scores of the output layer and
compares it to the ground truth labels of the data. The loss function yields a high value when the prediction doesn't
match the ground truth labels and a low value if otherwise.

[Adam](https://arxiv.org/abs/1412.6980) is used to update learnable parameters.

Image Classificaton
---
For the image classification task, the CIFAR-10 dataset is chosen. It contains 60000 images which have a size of
32 x 32 pixels. Each image is labelled as one of 10 classes. There are 6000 images per
class. Out of 60000 images, 50000 are used for training and 10000 are used for testing.

### Results
The objective is to predict the classes of the test dataset (10000 images) based on trained parameters using the 50000
training images. The current test accuracy is 46 %.

### Architecture
The neural network is a four layer network of the following structure:

`IN (n x 3072) -> H1 (4000 x 1) -> H2 (1000 x 1) -> H3 (4000 x 1) -> OUT (10 x 1)`

where `n` is the number of input observations (training images).

The size of the input layer `IN` and output layer `OUT` are fixed (as the amount of pixels and colors doesn't change).
The sizes of the hidden layers resulted from empirical observation and are not guaranteed to be fully optimized for the
task.

### Input layer
Input layer data is given as a vector consisting of rows for each input observation (image). Each image has 3 color
channels for each of the 32 x 32 pixels. This data is flattened out into a vector of length 3072 (3 x 32 x 32).

### Hidden layer
The network uses a fully connected architecture with hidden layers with ReLU as an activation function.

[Dropout](http://jmlr.org/papers/v15/srivastava14a.html) is used to prevent overfitting.

### Output layer
The output layer contains the class scores for each input observation. There are 10 scores, each representing a class.
The highest class score should be assigned to the correct class.

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
The image classification accuracy is clearly not enough for real-world scenarios. The problem is that a multi-layer 
perceptron is very expensive both in CPU and memory usage. That's why
[Convolutional neural networks](https://en.wikipedia.org/wiki/Convolutional_neural_network) were devised. These are
neural networks designed specifically for the task of image classification. Coding the neural network as convolutional
network would greatly increase performance.