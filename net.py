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


def calculate_activation(x, alpha):
    """
    Performs the activation function of a layer.
    This neural net uses leaky ReLU, which simply returns the input (and caps it at zero).
    :param x
    :return: The resulting matrix representing the activation values of the layer.
    """
    for i in np.where(x < 0)[1]:
        x[0][i] = alpha * x[0][i]
    return x


def data_loss(s, y, delta):
    """
    Calculates the data loss of the scores (how much the scores deviate from the ground truth labels).
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
    return .5 * lambda_ * np.sum(np.square(w))


def calculate_loss(s, y, w, delta, lambda_):
    """
    Calculates the loss of a score matrix depending on the ground truth labels. This method uses hinge loss (from MSVM).
    :param s: The score matrix of form (N x K), where N is the number of images and K is the number of classes.
    :param y: The ground truth label array of length N.
    :param w: The weight matrix of the output layer of form (H x K), where H is the size of the previous layer.
    :param delta: The data loss hyperparameter.
    :param lambda_: The regularization loss hyperparameter.
    :return: The hinge loss.
    """
    return data_loss(s, y, delta) + regularization_loss(w, lambda_)

def probs(scores):
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
from math import log
def calculate_cross_entropy_loss(s, y, w, lambda_):
    exp_scores = np.exp(s)
    p = 0
    for i in range(len(y)):
        sum_scores = np.sum(exp_scores[i][0])
        p += - log(exp_scores[i][0][y[i]] / sum_scores)
    data_loss = p / len(y)
    return data_loss + regularization_loss(w, lambda_)


def loss_gradient_by_scores(s, y, delta):
    """
    Calculates the gradient of the loss function by the scores.
    The gradient formula is: ds / dL = 1(s_j - s_y_i + delta > 0)
    :param s: The score parameter of the loss function.
    :param y: The ground truth label parameter of the loss function.
    :param delta: The data loss hyperparameter.
    :return: The gradient as a matrix of the same shape as `s`.
    """
    for i in range(len(y)):
        y_i = y[i]
        for j in range(len(s[i][0])):
            if j == y_i:
                class_score = s[i][0][j]
                count = s[i][0][s[i][0] - class_score + delta > 0].size
                s[i][0][j] = count
            elif s[i][0][j] - s[i][0][y_i] + delta < 0:
                s[i][0][j] = 0
            else:
                s[i][0][j] = 1
    return s

def cross_entropy_loss_gradient(s, y):
    exp_scores = np.exp(s)
    for i in range(len(y)):
        for j in range(len(exp_scores[i][0])):
            sum_scores = np.sum(exp_scores[i][0])
            exp_scores[i][0][j] = exp_scores[i][0][j] / sum_scores
            if j == y[i]:
                exp_scores[i][0][j] -= 1
    ds = exp_scores / len(y)
    return ds

def update_parameter(x, dx, epoch, m, v, learning_rate):
    """
    Updates the parameter x using its gradient dx.
    :param x: The parameter to be updated.
    :param dx: The gradient of the parameter to be updated.
    :param epoch: The current training epoch.
    :param m: The current momentum.
    :param v: The current velocity.
    :param learning_rate: The learning rate of the network.
    :return:
    """
    m = beta1 * m + (1 - beta1) * dx
    mt = m / (1 - beta1 ** epoch)
    v = beta2 * v + (1 - beta2) * (np.square(dx))
    vt = v / (1 - beta2 ** epoch)
    update = - learning_rate * mt / (np.sqrt(vt) + eps)
    #print(np.mean(update / x))
    x += update


# hyperparameters
delta = 1  # Data loss parameter
lambda_ = 0  # The regularization strength (has an influence on regularization loss).
learning_rate = .1  # The step size for each epoch (influences how greedy the network changes its parameters)
epochs = 1000  # The amount of iterations the network should take
alpha = .0  # Slope for leaky ReLU
beta1 = .9  # Hyperparameter for Adam parameter update
beta2 = .999  # Hyperparameter for Adam parameter update
eps = 1e-8  # Hyperparameter for Adam parameter update

# Input data: 80 % train, 10 % val, 10 % test
x_data, y_data = get_data()

x_tr, x_val, x_te = split_data(x_data, .8, .1, .1)
y_tr, y_val, y_te = split_data(y_data, .8, .1, .1)

# Preprocess data
x_tr, x_val, x_te, pre_mean, pre_std = preprocess_data(x_tr, x_val, x_te)

# Neural net: IN (3072 x 1) -> HL (1000 x 1) -> HL (100 x 1) -> OUT (10 x 1)
k = len(np.unique(y_tr))  # number of classes
hidden_sizes = [1000, 500, 25]
out_size = k

# Quick accuracy measure
# cc = 0
# with open('dump.p', 'rb') as file:
#     wh, wo, bh, bo = load(file)
#     for i in range(len(x_te)):
#         input_layer = x_te[i]
#         hiddens = []
#         for h in range(len(wh)):
#             if h == 0:
#                 hidden = input_layer.dot(wh[h]) + bh[h]
#             else:
#                 hidden = hiddens[h - 1].dot(wh[h]) + bh[h]
#             hidden = calculate_activation(hidden, alpha)
#             hiddens.append(hidden)
#         out = hiddens[-1].dot(wo) + bo
#         if np.argmax(out) == y_te[i]:
#             cc += 1
# print(cc / len(x_te))

# weight initialization (specialized for relu activations)
wh = []
bh = []
for i in range(len(hidden_sizes)):
    if i == 0:
        wh.append(np.random.randn(x_tr[0].shape[0], hidden_sizes[i]) * sqrt(2 / x_tr[0].shape[0]))
    else:
        wh.append(np.random.randn(hidden_sizes[i - 1], hidden_sizes[i]) * sqrt(2 / x_tr[0].shape[0]))
    bh.append(np.zeros((1, hidden_sizes[i])))
w_out = np.random.randn(hidden_sizes[-1], out_size) * sqrt(2 / x_tr[0].shape[0])
b_out = np.zeros((1, out_size))
m = .0
v = .0

for epoch in range(1, epochs):
    outs = []
    hidden_layers = [[] for h in hidden_sizes]
    deads = 0
    total = 0
    out_deads = 0
    # forward pass
    for input_nr in range(len(x_tr)):
        input_layer = x_tr[input_nr]
        # for h in range(len(hidden_sizes)):
        #     if h == 0:
        #         hidden = np.dot(wh[h], input_layer) + bh[h]
        #         hidden = calculate_activation(hidden, alpha)
        #         hidden_layers[h].append(hidden)
        #     else:
        #         hidden = wh[h].dot(hidden_layers[h - 1][input_nr]) + bh[h]
        #         hidden = calculate_activation(hidden, alpha)
        #         hidden_layers[h].append(hidden)
        for h in range(len(hidden_sizes)):
            if h == 0:
                hidden = input_layer.dot(wh[h]) + bh[h]
                hidden = calculate_activation(hidden, alpha)
                hidden_layers[h].append(hidden)
            else:
                hidden = hidden_layers[h - 1][input_nr].dot(wh[h]) + bh[h]
                hidden = calculate_activation(hidden, alpha)
                hidden_layers[h].append(hidden)

        # hidden1 = input_layer.dot(w_hidden1) + b_hidden1
        # hidden1 = calculate_activation(hidden1, alpha)
        # deads += hidden1[hidden1 == 0].size
        # total += hidden1.size
        # hiddens_1.append(hidden1)
        # # ToDo: Apply dropout
        # hidden2 = hidden1.dot(w_hidden2) + b_hidden2
        # hidden2 = calculate_activation(hidden2, alpha)
        # deads += hidden2[hidden2 == 0].size
        # total += hidden2.size
        # hiddens_2.append(hidden2)
        out = hidden_layers[-1][input_nr].dot(w_out) + b_out
        out_deads += out[out == 0].size
        outs.append(out)

    # Calculate loss
    loss = calculate_loss(outs, y_tr, w_out, delta, lambda_)
    print(epoch, loss, deads, total, out_deads)

    # Backpropagation

    dscores = loss_gradient_by_scores(outs, y_tr, delta)
    dwh = [np.full(w_i.shape, 1.0) for w_i in wh]
    dbh = [np.empty(b_i.shape) for b_i in bh]
    dw_out = np.full(w_out.shape, 1.0)
    db_out = np.empty(b_out.shape)

    dws = [[] for w_i in wh]
    dos = []

    for i in range(len(hidden_layers[0])):
        ds = dscores[i]
        #dw_out *= hidden_layers[-1][i].T.dot(ds)
        dos.append(hidden_layers[-1][i].T.dot(ds))
        db_out += np.sum(ds, axis=0, keepdims=True)
        dhiddens = {}
        for h in range(len(hidden_layers) - 1, -1, -1):
            if h == len(hidden_layers) - 1:
                dhidden = ds.dot(w_out.T)
            else:
                dhidden = dhiddens[h + 1].dot(wh[h + 1].T)
            dhidden[hidden_layers[h][i] < 0] = alpha
            dhiddens[h] = dhidden
            if h == 0:
                #dwh[h] *= x_tr[i].reshape(1, 3072).T.dot(dhidden)
                dws[h].append(x_tr[i].reshape(1, 3072).T.dot(dhidden))
            else:
                #dwh[h] *= hidden_layers[h - 1][i].T.dot(dhidden)
                dws[h].append(hidden_layers[h - 1][i].T.dot(dhidden))
            dbh[h] += np.sum(dhidden, axis=0, keepdims=True)

    dw_out += lambda_ * w_out

    # Set weights using gradients of backpropagation
    for h in range(len(hidden_layers)):
        update_parameter(wh[h], np.array(dws).dot(x_tr), epoch, m, v, learning_rate)
        update_parameter(bh[h], dbh[h], epoch, m, v, learning_rate)
    update_parameter(w_out, np.array(dos).dot(x_tr), epoch, m, v, learning_rate)
    update_parameter(b_out, db_out, epoch, m, v, learning_rate)

with open('dump.p', 'wb') as dump_file:
    dump((wh, w_out, bh, b_out), dump_file)
