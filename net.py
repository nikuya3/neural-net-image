import numpy as np
from pickle import load
from random import sample
from math import sqrt

def unpickle(file):
    with open(file, 'rb') as fo:
        data = load(fo, encoding='bytes')
    return data


def get_data():
    """
    Reads input data from the 'data' directory.
    :return: A matrix of form 5000 x 3072
    """
    x_data = None
    y_data = []
    for nr in range(1, 6):
        batch = unpickle('data/data_batch_' + str(nr))
        if x_data is None:
            x_data = batch[b'data']
            y_data = batch[b'labels']
        else:
            x_data = np.concatenate((x_data, batch[b'data']), axis=0)
            y_data += batch[b'labels']
    return x_data, y_data

def split_data(data, train, val, test):
    """
    Splits the given data into a training, a validation and a test set.
    :param data: The data to be splitted.
    :param train: A percentage indicating the size of the training set.
    :param val: A percentage indicating the size of the validation set.
    :param test: A percentage indicating the size of the test set.
    :return: The training, validation and test sets
    """
    training_set = data[:int(len(data) * train)]
    val_end = int(len(training_set) + len(data) * val)
    val_set = data[len(training_set):val_end]
    test_end = int(len(training_set) + len(val_set) + len(data) * test)
    test_set = data[val_end:test_end]
    return training_set, val_set, test_set

def preprocess_data(train, validation, test):
    """
    Applies zero-centering and normalization of input data for better model performance.
    :param train:
    :param validation:
    :param test:
    :return:
    """
    train = train.astype(float)
    validation = validation.astype(float)
    test = test.astype(float)
    # Zero-centering (subtracting the mean)
    mean = np.mean(train, axis=0)  # Using statistic of training set
    train -= mean
    validation -= mean
    test -= mean
    # Normalization of data dimension to be of equal scale (division by standard deviation)
    std = np.std(train, axis=0)
    train /= std
    validation /= std
    test /= std
    return train, validation, test, mean, std

def calculate_activation(x):
    """
    Performs the activation function of a layer.
    This neural net uses ReLU, which simply returns the input (and caps it at zero).
    :param x
    :return: The resulting matrix representing the activation values of the layer.
    """
    x[x < 0] = 0
    return x

# hyperparameters
learning_rate = .01
regularization_strength = .01

# Input data: 80 % train, 10 % val, 10 % test
x_data, y_data = get_data()

x_tr, x_val, x_te = split_data(x_data, .8, .1, .1)
y_tr, y_val, y_te = split_data(y_data, .8, .1, .1)

# Preprocess data
x_tr, x_val, x_te, pre_mean, pre_std = preprocess_data(x_tr, x_val, x_te)

# Neural net: IN (3072 x 1) -> HL (100 x 100) -> HL (100 x 1) -> OUT (10 x 1)
input_layer = x_tr[0]

k = len(np.unique(y_data))  # number of classes
hidden1_shape = [3072, 100]
hidden2_shape = [100, 100]
hidden_shapes = [hidden1_shape, hidden2_shape]
out_shape = [hidden2_shape[0], k]

# weight initialization (specialized for relu activations)
w_hidden1 = np.random.randn(hidden1_shape[0], hidden1_shape[1]) * sqrt(2 / input_layer.shape[0])
w_hidden2 = np.random.randn(hidden2_shape[0], hidden2_shape[1]) * sqrt(2 / w_hidden1.shape[1])
w_out = np.random.randn(out_shape[0], out_shape[1]) * sqrt(2 / w_hidden2.shape[1])

# forward pass
hidden1 = input_layer.dot(w_hidden1)
hidden1 = calculate_activation(hidden1)
hidden2 = hidden1.dot(w_hidden2)
hidden2 = calculate_activation(hidden2)
out = hidden2.dot(w_out)

# ToDo: Loss function

# ToDo: Backpropagation

# ToDo: Set weights (incrementally considering learning rate)