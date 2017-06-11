import numpy as np
from pickle import dump, load
from math import log, sqrt


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


def initialize_parameters(input_size, hidden_sizes, output_size):
    """
    Initializes the learnable parameters for a neural network (basically the connection weights and biases).
    The parameters are designed to work well with ReLU.
    :param input_size: The size of the input layer.
    :param hidden_sizes: The hidden layer sizes as array.
    :param output_size: The size of the output layer.
    :return: The connection weights and biases for the neural networks.
    """
    wh = []
    bh = []
    for i in range(len(hidden_sizes)):
        if i == 0:
            wh.append(np.random.randn(input_size, hidden_sizes[i]) * sqrt(2 / input_size))
        else:
            wh.append(np.random.randn(hidden_sizes[i - 1], hidden_sizes[i]) * sqrt(2 / input_size))
        bh.append(np.zeros((1, hidden_sizes[i])))
    w_out = np.random.randn(hidden_sizes[-1], output_size) * sqrt(2 / input_size)
    b_out = np.zeros((1, output_size))
    return wh, bh, w_out, b_out


def calculate_activation(x, alpha):
    """
    Performs the activation function of a layer.
    This neural net uses leaky ReLU. This activation returns the input if the input is positive.
    If the input is negative, it returns the input multiplied by alpha.
    In usual leaky ReLU environments, alpha should by a small number, so that the output is nearly 0.
    For a normal ReLU behaviour, simply pass alpha as 0.
    :param x: The input matrix to the activation.
    :param alpha: The factor by which negative inputs are scaled.
    :return: The resulting matrix representing the activation values of the layer.
    """
    for i in np.where(x < 0)[1]:
        x[0][i] = alpha * x[0][i]
    return x


def forward_pass(x, hidden_sizes, out_size, wh, bh, w_out, b_out, alpha):
    """
    Performs the forward pass of a neural network.
    :param x: The input data of form (N x D), where N is the number of observations an D is the dimensionality.
    :param hidden_sizes: The size of each hidden layer as array.
    :param out_size: The size of the output layer.
    :param wh: The weights of each hidden layer connection as array. Each weight is a matrix of (H_i-1 ... H_i),
    where H_i-1 is the size of the previous hidden layer (or the input layer) and H_i is the size of the corresponding
    hidden layer..
    :param bh: The biases of each hidden layer as array. Each bias is a vector of the same length of the corresponding
    hidden layer.
    :param w_out: The weight of the output layer as matrix of form (H x out_size),
    where H is the size of the last hidden layer.
    :param b_out: The bias of the output layer as vector of length out_size.
    :param alpha: The factor by which negative inputs are scaled in ReLU activations. Set to 0 to avoid leaky ReLU.
    :return: A tuple consisting of the following values:
    * An array containing the values of each hidden layer as vector of length hidden_size[i] for every input observation.
    * An array containing the class scores of each input observation.
    * The connection weights of the last layer (output_layer).
    """
    outs = np.empty((len(x), out_size))
    hidden_layers = [np.empty((len(x), size)) for size in hidden_sizes]
    for input_nr in range(len(x)):
        input_layer = x[input_nr]
        for h in range(len(hidden_sizes)):
            if h == 0:
                hidden = input_layer.dot(wh[h]) + bh[h]
                hidden = calculate_activation(hidden, alpha)
                hidden_layers[h][input_nr] = hidden
            else:
                hidden = hidden_layers[h - 1][input_nr].dot(wh[h]) + bh[h]
                hidden = calculate_activation(hidden, alpha)
                hidden_layers[h][input_nr] = hidden
        # ToDo: Apply dropout
        out = hidden_layers[-1][input_nr].dot(w_out) + b_out
        outs[input_nr] = out
    return hidden_layers, outs, w_out


def data_hinge_loss(s, y, delta):
    """
    Calculates the data loss of the scores (how much the scores deviate from the ground truth labels).
    :param s: The score matrix of form (N x K), where N is the number of images and K is the number of classes.
    :param y: The ground truth label array of length N.
    :param delta: A hyperparameter which indicates the minimum margin between a score and the score of the ground truth
    label.
    :return: The data loss.
    """
    loss = 0
    for i in range(len(y)):
        y_i = y[i]
        for j in range(len(s[i][0])):
            if j != y_i:
                loss += max(0, s[i][0][j] - s[i][0][y_i] + delta)
    return loss / len(s)


def regularization_loss(w, lambda_):
    """
    Calculates the regularization loss of the output weights.
    Regularization loss is used to favor smaller magnitudes of weights.
    :param w: The weight matrix of the output layer of form (H x K),
    where H is the size of the previous layer and K is the number of classes.
    :param lambda_: A hyperparameter used to control the magnitude of the weight.
    :return: The regularization loss.
    """
    return .5 * lambda_ * np.sum(np.square(w))


def calculate_hinge_loss(s, y, w, delta, lambda_):
    """
    Calculates the loss of a score matrix depending on the ground truth labels. This method uses hinge loss (from MSVM).
    :param s: The score matrix of form (N x K), where N is the number of images and K is the number of classes.
    :param y: The ground truth label vector of length N.
    :param w: The weight matrix of the output layer of form (H x K), where H is the size of the previous layer.
    :param delta: The data loss hyperparameter.
    :param lambda_: The regularization loss hyperparameter.
    :return: The hinge loss.
    """
    return data_hinge_loss(s, y, delta) + regularization_loss(w, lambda_)


def probs(scores):
    """
    Calculates the probabilities out of a neural networks class scores.
    :param scores: The score matrix of form (N x K), where N is the number of observations and K is the number of classes.
    :return: The probabilities of the same form as the input scores.
    """
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def calculate_cross_entropy_loss(s, y, w, lambda_):
    """
    Calculates the loss of a score matrix depending on the ground truth labels.
    This method uses cross entropy loss (from Softmax).
    :param s: The score matrix of form (N x K), where N is the number of observations and K is the number of classes.
    :param y: The ground truth label vector of length N.
    :param w: The weight matrix of the output layer of form (H x K), where H is the size of the previous layer.
    :param lambda_: The regularization loss hyperparameter.
    :return: The cross-entropy loss, where 0 indicates a perfect match between s and y
    and +Inf indicates a perfect mismatch.
    """
    probabilities = probs(s)
    log_probabilities = - np.log(probabilities[range(len(y)), y])
    data_loss = np.sum(log_probabilities) / len(y)
    return data_loss + regularization_loss(w, lambda_)


def hinge_loss_gradient_by_scores(s, y, delta):
    """
    Calculates the gradient of the hinge loss function by the scores.
    The gradient formula is: { ds_j / dL = 1(s_j - s_y_i + delta > 0), ds_y_i / dL = sum (1(s_j - s_y_i + delta > 0) }
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
    """
    Calculates the gradient of the hinge loss function by the scores.
    The gradient formula is { ds_j / dL = e^s_j / sum e^j, ds_y_i / dL = e^s_y_i / sum e^j - 1 }.
    :param s: The score parameter of the loss function.
    :param y: The ground truth label parameter of the loss function.
    :return: The gradient as a matrix of the same shape as `s`.
    """
    dscores = s
    s[range(len(y)), y] -= 1
    dscores /= len(y)
    return dscores


def backpropagation(x, s, y, hidden_layers, wh, bh, w_out, b_out, alpha):
    """
    Performs the backpropagation of a neural network.
    :param x: The input data of form (N x D), where N is the number of observations an D is the dimensionality.
    :param s: The score matrix of form (N x K), where N is the number of observations and K is the number of classes.
    :param y: The ground truth labels for each observation.
    :param hidden_layers: An array containing the values of each hidden layer as a vector.
    :param wh: The weights of each hidden layer connection as array. Each weight is a matrix of (H_i-1 ... H_i),
    where H_i-1 is the size of the previous hidden layer (or the input layer) and H_i is the size of the corresponding
    hidden layer..
    :param bh: The biases of each hidden layer as array. Each bias is a vector of the same length of the corresponding
    hidden layer.
    :param w_out: The weight of the output layer as matrix of form (H x K),
    where H is the size of the last hidden layer and K is the number of classes.
    :param b_out: The bias of the output layer as vector of length K, where K is the number of classes.
    :param alpha: The factor by which negative inputs are scaled in ReLU activations. Set to 0 to avoid leaky ReLU.
    :return: The backpropagation returns relevant gradients as a tuple containing the following values:
    * An array containing the gradients for the connection weights of each hidden layer of the same form as `wh`.
    * An array containing the gradients for the biases of each hidden layer of the same form as `bh`.
    * An array containing the gradients for the connection weights of the output layer of the same form as `w_out`.
    * An array containing the gradients for the biases of the output layer of the same form as `b_out`.
    """
    dscores = cross_entropy_loss_gradient(s, y)
    dwh = [np.full(w_i.shape, .0) for w_i in wh]
    dbh = [np.empty(b_i.shape) for b_i in bh]
    dw_out = np.full(w_out.shape, .0)
    db_out = np.empty(b_out.shape)

    for i in range(len(hidden_layers[0])):
        ds = dscores[i]
        dw_out += np.asmatrix(hidden_layers[-1][i]).T.dot(np.asmatrix(ds))
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
                dwh[h] += np.asmatrix(x_tr[i]).T.dot(np.asmatrix(dhidden))
            else:
                dwh[h] += np.asmatrix(hidden_layers[h - 1][i]).T.dot(np.asmatrix(dhidden))
            dbh[h] += np.sum(dhidden, axis=0, keepdims=True)
    dw_out += lambda_ * w_out
    return dwh, dbh, dw_out, db_out


def update_parameter(x, dx, epoch, learning_rate, m, v, beta1, beta2, eps):
    """
    Updates the parameter x using its gradient dx.
    :param x: The parameter to be updated.
    :param dx: The gradient of the parameter to be updated.
    :param epoch: The current training epoch.
    :param learning_rate: The learning rate of the network. Indicates the size of learning steps.
    :param m: The current momentum.
    :param v: The current velocity.
    :param beta1: Hyperparameter for the Adam parameter update. Recommended to be .9.
    :param beta2: Hyperparameter for the Adam parameter update. Recommended to be .999.
    :param eps: Hyperparameter for the Adam parameter update. Recommended to be 1e-8.
    :return: The updated parameter of the same type as x.
    """
    m = beta1 * m + (1 - beta1) * dx
    mt = m / (1 - beta1 ** epoch)
    v = beta2 * v + (1 - beta2) * (np.square(dx))
    vt = v / (1 - beta2 ** epoch)
    update = - learning_rate * mt / (np.sqrt(vt) + eps)
    #print(np.mean(update / x))
    x += update


def train(epochs, wh, bh, w_out, b_out, learning_rate, alpha, beta1, beta2, eps, lambda_):
    """
    Trains a neural network. The learnable parameters `wh`, `bh`, `w_out` and `b_out` are optimized as long as
    `epochs` indicates and then returned.
    :param epochs: The amount of runs a network should take. (Note: The number of iterations is 2 * N * epochs,
    where N is the number of inputs.)
    :param wh: The initialized weights of each hidden layer connection as array. Each weight is a matrix of (H_i-1 ... H_i),
    where H_i-1 is the size of the previous hidden layer (or the input layer) and H_i is the size of the corresponding
    hidden layer..
    :param bh: The initialized biases of each hidden layer as array. Each bias is a vector of the same length of the corresponding
    hidden layer.
    :param w_out: The initialized weight of the output layer as matrix of form (H x K),
    where H is the size of the last hidden layer and K is the number of classes.
    :param b_out: The initialized bias of the output layer as vector of length K, where K is the number of classes.
    :param learning_rate: Indicates the step size of each learning epoch. High values lead to faster trainings,
    but also inhibit the risk of overstepping. Low values take longer to train, but will eventually reach the desired
    effect.
    :param alpha: The factor by which negative inputs are scaled in ReLU activations. Set to 0 to avoid leaky ReLU.
    :param beta1: Hyperparameter for the Adam parameter update. Recommended to be .9.
    :param beta2: Hyperparameter for the Adam parameter update. Recommended to be .999.
    :param eps: Hyperparameter for the Adam parameter update. Recommended to be 1e-8.
    :return: A tuple containing the optimized learnable parameters `wh`, `bh`, `w_out` and `b_out`.
    """
    m = .0
    v = .0
    for epoch in range(1, epochs + 1):
        # Feed-forward the network
        hidden_layers, outs, w_out = forward_pass(x_tr, hidden_sizes, out_size, wh, bh, w_out, b_out, alpha)

        # Calculate loss
        loss = calculate_cross_entropy_loss(outs, y_tr, w_out, lambda_)
        print(epoch, loss)

        # Backpropagation
        dwh, dbh, dw_out, db_out = backpropagation(x_tr, outs, y_tr, hidden_layers, wh, bh, w_out, b_out, alpha)

        # Update parameters using gradients of backpropagation
        for h in range(len(hidden_layers)):
            update_parameter(wh[h], dwh[h], epoch, learning_rate, m, v, beta1, beta2, eps)
            update_parameter(bh[h], dbh[h], epoch, learning_rate, m, v, beta1, beta2, eps)
        update_parameter(w_out, dw_out, epoch, learning_rate, m, v, beta1, beta2, eps)
        update_parameter(b_out, db_out, epoch, learning_rate, m, v, beta1, beta2, eps)
    return wh, bh, w_out, b_out


def predict(x, hidden_sizes, out_size, wh, bh, w_out, b_out, alpha):
    """
    Predicts the classes of the given input observations.
    :param x: The input observations to classify.
    :param hidden_sizes: The size of each hidden layer as array.
    :param out_size: The size of the output layer.
    :param wh: The weights of each hidden layer connection as array. Each weight is a matrix of (H_i-1 ... H_i),
    where H_i-1 is the size of the previous hidden layer (or the input layer) and H_i is the size of the corresponding
    hidden layer..
    :param bh: The biases of each hidden layer as array. Each bias is a vector of the same length of the corresponding
    hidden layer.
    :param w_out: The weight of the output layer as matrix of form (H x out_size),
    where H is the size of the last hidden layer.
    :param b_out: The bias of the output layer as vector of length out_size.
    :param alpha: The factor by which negative inputs are scaled in ReLU activations. Set to 0 to avoid leaky ReLU.
    :return: The indices of the correct classes.
    """
    _, out, _ = forward_pass(x, hidden_sizes, out_size, wh, bh, w_out, b_out, alpha)
    return np.argmax(out, axis=1)

def accuracy(x, y, hidden_sizes, out_size, wh, wo, bh, bo, alpha):
    """
    Measures the accuracy of a nerual network. Specifically the proportion of correct predictions of input data x
    using the parameters wh, wo, bh, bo and the ground truth labels y.
    :param x: The input data to be predicted.
    :param y: The ground truth labels for the input data.
    :param hidden_sizes: The size of each hidden layer as array.
    :param out_size: The size of the output layer.
    :param wh: The weights of each hidden layer as an array.
    :param wo: The weights of the output layer.
    :param bh: The biases of each hidden layer as an array.
    :param bo: The biases of the output layer.
    :param alpha: The factor by which negative inputs are scaled in ReLU activations. Set to 0 to avoid leaky ReLU.
    :return: The accuracy as proportion, where 1 indicates a perfect match and 0 indicates a perfect mismatch.
    """
    predicted_classes = predict(x, hidden_sizes, out_size, wh, bh, wo, bo, alpha)
    correct_classes = len(np.where(predicted_classes == y))
    return correct_classes / len(x)

# hyperparameters
delta = 1  # The minimum margin of the hinge loss
lambda_ = 0.1  # The regularization strength (has an influence on regularization loss).
learning_rate = .001  # The step size for each epoch (influences how greedy the network changes its parameters)
epochs = 100  # The amount of 'iterations' the network should take
alpha = .0  # Slope for leaky ReLU. Set to 0 to avoid leaky ReLU.
beta1 = .9  # Hyperparameter for Adam parameter update
beta2 = .999  # Hyperparameter for Adam parameter update
eps = 1e-8  # Hyperparameter for Adam parameter update

# Input data: 80 % train, 10 % val, 10 % test
x_data, y_data = get_data()

x_tr, x_val, x_te = split_data(x_data, .8, .1, .1)
y_tr, y_val, y_te = split_data(y_data, .8, .1, .1)

# Preprocess data
x_tr, x_val, x_te, pre_mean, pre_std = preprocess_data(x_tr, x_val, x_te)

# Neural net: IN (3072 x 1) -> HL (1000 x 1) -> HL (500 x 1) -> HL (24 x 1) -> OUT (10 x 1)
k = len(np.unique(y_tr))  # number of classes
n = len(x_tr)  # number of inputs
hidden_sizes = [1000, 500, 25]
out_size = k

# Parameter initialization
wh, bh, w_out, b_out = initialize_parameters(x_tr[0].shape[0], hidden_sizes, out_size)

# Train the network
train(epochs, wh, bh, w_out, b_out, learning_rate, alpha, beta1, beta2, eps, lambda_)

# Save parameters for reuse
with open('dump.p', 'wb') as dump_file:
    dump((wh, w_out, bh, b_out), dump_file)

# Quick accuracy
with open('dump.p', 'rb') as file:
    wh, wo, bh, bo = load(file)
    print(accuracy(x_val, y_val, hidden_sizes, out_size, wh, wo, bh, bo, alpha))
