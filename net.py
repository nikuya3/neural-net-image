import numpy as np
from pickle import dump, load
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
    return x_data[80:120], y_data[80:120]


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

indices = []


def calculate_activation(x):
    """
    Performs the activation function of a layer.
    This neural net uses ReLU, which simply returns the input (and caps it at zero).
    :param x
    :return: The resulting matrix representing the activation values of the layer.
    """
    x[x < 0] = 0
    return x


def data_loss(s, y, delta):
    """
    Calculates the data loss of the scores (how much the scores deviate from the ground truth labels.
    :param s: The score matrix of form (N x K), where N is the number of images and K is the number of classes.
    :param y: The ground truth label array of length N.
    :param delta: A hyperparameter which indicates the minimum difference between a score and the score of the ground truth label.
    :return: The data loss.
    """
    loss = 0
    # ToDo: Convert iterator to array operation
    #x = np.sum(np.maximum(0, s - s[y] + delta))
    for i in range(len(y)):
        y_i = y[i]
        for j in range(len(s[i][0])):
            if j != y_i:
                loss += max(0, s[i][0][j] - s[i][0][y_i] + delta)
    return loss / len(s)


def regularization_loss(w, lambda_):
    """
    Calculates the regularization loss of the output weights. Regularization loss is used to favor smaller magnitudes of weights.
    :param w: The weight matrix of the output layer of form (H x K), where H is the size of the previous layer and K is the number of classes.
    :param lambda_: A hyperparameter used to control the magnitude of the weight.
    :return: The regularization loss.
    """
    return lambda_ * np.sum(np.square(w))


def calculate_loss(s, y, w, delta, lambda_):
    """
    Calculates the loss of a score matrix depending on the ground truth labels. This method uses multiclass SVM loss.
    :param s: The score matrix of form (N x K), where N is the number of images and K is the number of classes.
    :param y: The ground truth label array of length N.
    :param w: The weight matrix of the output layer of form (H x K), where H is the size of the previous layer.
    :param delta: The data loss hyperparameter.
    :param lambda_: The regularization loss hyperparameter.
    :return: The MSVM loss.
    """
    return data_loss(s, y, delta) + regularization_loss(w, lambda_)


def loss_gradient_by_scores(s, y, delta):
    """
    Calculates the gradient of the loss function by the scores.
    The gradient formula is: ds / dL = 1(0, s_j - s_y_i + delta)
    :param s: The score parameter of the loss function.
    :param y: The ground truth label parameter of the loss function.
    :param delta: The data loss hyperparameter.
    :return: The gradient as a matrix of the same shape as `s`.
    """
    for i in range(len(y)):
        y_i = y[i]
        for j in range(len(s[i][0])):
            if s[i][0][j] - s[i][0][y_i] + delta < 0:
                s[i][0][j] = 0
            else:
                s[i][0][j] = 1
    return s


# hyperparameters
delta = 1  # Data loss parameter
lambda_ = 0  # The regularization strength (has an influence on regularization loss).
learning_rate = .01  # The step size for each epoch (influences how greedy the network changes its parameters)
epochs = 100  # The amount of iterations the network should take

# Input data: 80 % train, 10 % val, 10 % test
x_data, y_data = get_data()

x_tr, x_val, x_te = split_data(x_data, .8, .1, .1)
y_tr, y_val, y_te = split_data(y_data, .8, .1, .1)

# Preprocess data
x_tr, x_val, x_te, pre_mean, pre_std = preprocess_data(x_tr, x_val, x_te)

# Neural net: IN (3072 x 1) -> HL (100 x 100) -> HL (100 x 1) -> OUT (10 x 1)
k = len(np.unique(y_data))  # number of classes
hidden1_shape = [1000, 1000]
hidden2_shape = [100, 1]
hidden_shapes = [hidden1_shape, hidden2_shape]
out_shape = [k, 1]

# weight initialization (specialized for relu activations)
w_hidden1 = np.random.randn(x_tr[0].shape[0], hidden1_shape[0]) * sqrt(2 / x_tr[0].shape[0])
w_hidden2 = np.random.randn(hidden1_shape[1], hidden2_shape[0]) * sqrt(2 / x_tr[0].shape[0])
w_out = np.random.randn(hidden2_shape[0], out_shape[0]) * sqrt(2 / x_tr[0].shape[0])

b_hidden1 = np.zeros((1, hidden1_shape[0]))
b_hidden2 = np.zeros((1, hidden2_shape[0]))
b_out = np.zeros((1, out_shape[0]))

for epoch in range(epochs):
    outs = []
    hiddens_1 = []
    hiddens_2 = []
    # forward pass
    for row in x_tr:
        input_layer = row
        hidden1 = input_layer.dot(w_hidden1) + b_hidden1
        hidden1 = calculate_activation(hidden1)
        hiddens_1.append(hidden1)
        # ToDo: Apply dropout
        hidden2 = hidden1.dot(w_hidden2) + b_hidden2
        hidden2 = calculate_activation(hidden2)
        hiddens_2.append(hidden2)
        out = hidden2.dot(w_out) + b_out
        outs.append(out)


    # Calculate loss
    loss = calculate_loss(outs, y_tr, w_out, delta, lambda_)
    print(epoch, loss)

    # Backpropagation

    dscores = loss_gradient_by_scores(outs, y_tr, delta)
    dw_out = np.empty(w_out.shape)
    db_out = np.empty(b_out.shape)
    dw_hidden1 = np.empty(w_hidden1.shape)
    db_hidden1 = np.empty(b_hidden1.shape)
    dw_hidden2 = np.empty(w_hidden2.shape)
    db_hidden2 = np.empty(b_hidden2.shape)

    for i in range(len(hiddens_1)):
        ds = dscores[i]
        dw_out += hiddens_2[i].T.dot(ds)
        db_out += np.sum(ds, axis=0, keepdims=True)
        dhidden2 = ds.dot(w_out.T)
        dhidden2[hiddens_2[i] < 0] = 0

        dw_hidden2 += hiddens_1[i].T.dot(dhidden2)
        db_hidden2 += np.sum(dhidden2, axis=0, keepdims=True)
        dhidden1 = dhidden2.dot(w_hidden2.T)
        dhidden1[hiddens_1[i] < 0] = 0

        dw_hidden1 += x_tr[i].reshape(1, 3072).T.dot(dhidden1)
        db_hidden1 += np.sum(dhidden1, axis=0, keepdims=True)

    # Set weights using gradients of backpropagation

    w_hidden1 += - learning_rate * dw_hidden1 / len(hiddens_1)
    w_hidden2 += - learning_rate * dw_hidden2 / len(hiddens_1)
    w_out += - learning_rate * dw_out / len(hiddens_1)
    b_hidden1 += - learning_rate * db_hidden1 / len(hiddens_1)
    b_hidden2 += - learning_rate * db_hidden2 / len(hiddens_1)
    b_out += - learning_rate * db_out / len(hiddens_1)

with open('dump.p', 'wb') as dump_file:
    dump(w_hidden1, dump_file)
    dump(w_hidden2, dump_file)
    dump(w_out, dump_file)
    dump(b_hidden1, dump_file)
    dump(b_hidden2, dump_file)
    dump(b_out, dump_file)
