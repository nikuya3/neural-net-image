import numpy as np
from pickle import dump, load
from math import sqrt


def write_list(list, filename):
    """
    Writes a list to a file.
    :param list: The list to be written.
    :param filename: The name of the new file.
    """
    with open(filename, 'w') as file:
        for item in list:
            file.write(str(item) + '\n')


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


def get_batches(x, y, batch_size):
    """
    Gets a random subset of arrays x and y.
    :param x: An array to use.
    :param y: Another array to use.
    :param batch_size: The size of the subset.
    :return: A random subset of arrays x and y.
    """
    if batch_size > 0:
        random_indices = np.random.randint(x.shape[0], size=batch_size)
        batch_x = x[random_indices, :]
        batch_y = [y[i] for i in random_indices]
    else:
        batch_x = x
        batch_y = y
    return batch_x, batch_y


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
    x[x < 0] *= alpha
    return x


def activation_gradient(dx, x, alpha):
    """
    Calculates the gradient of the activation function in the backpropagation process. This network uses (leaky) ReLU.
    :param dx: The preceding gradient of the input matrix.
    :param x: The activated input matrix of the feedforward process.
    :param alpha: The factor by which negative inputs are scaled in leaky ReLU. Set to 0 to avoid leaky ReLU.
    :return: The gradient dx enriched by the gradient of the activation.
    """
    dx[x < 0] = alpha
    return dx


def forward_pass(x, hidden_sizes, wh, bh, w_out, b_out, alpha, p):
    """
    Performs the forward pass of a neural network.
    :param x: The input data of form (N x D), where N is the number of observations an D is the dimensionality.
    :param hidden_sizes: The size of each hidden layer as array.
    :param wh: The weights of each hidden layer connection as array. Each weight is a matrix of (H_i-1 ... H_i),
    where H_i-1 is the size of the previous hidden layer (or the input layer) and H_i is the size of the corresponding
    hidden layer..
    :param bh: The biases of each hidden layer as array. Each bias is a vector of the same length of the corresponding
    hidden layer.
    :param w_out: The weight of the output layer as matrix of form (H x out_size),
    where H is the size of the last hidden layer.
    :param b_out: The bias of the output layer as vector of length out_size.
    :param alpha: The factor by which negative inputs are scaled in ReLU activations. Set to 0 to avoid leaky ReLU.
    :param p: The probability of each neuron to be dropped out. Set to 1 to disable dropout.
    :return: A tuple consisting of the following values:
    * An array containing the values of each hidden layer as vector of length hidden_size[i] for every input observation.
    * An array containing the class scores of each input observation.
    * The connection weights of the last layer (output_layer).
    """
    hidden_layers = [np.empty((len(x), size)) for size in hidden_sizes]
    for h in range(len(hidden_sizes)):
        if h == 0:
            hidden = x.dot(wh[h]) + bh[h]
        else:
            hidden = hidden_layers[h - 1].dot(wh[h]) + bh[h]
        hidden = calculate_activation(hidden, alpha)
        dropout_mask = (np.random.random(hidden.shape) < p) / p
        hidden *= dropout_mask
        hidden_layers[h] = hidden
    outs = hidden_layers[-1].dot(w_out) + b_out
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
        for j in range(len(s[i])):
            if j != y_i:
                loss += max(0, s[i][j] - s[i][y_i] + delta)
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


def regularization_loss_gradient_by_weights(w, lambda_):
    """
    Calculates the gradient of the regularization loss function by the weights.
    :param w: The weight matrix of the output layer of form (H x K),
    where H is the size of the previous layer and K is the number of classes.
    :param lambda_: A hyperparameter used to control the magnitude of the weight.
    :return: The gradient.
    """
    return lambda_ * w


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
    dscores = probs(s)
    dscores[range(len(y)), y] -= 1
    dscores /= len(y)
    return dscores


def calculate_accuracy(scores, y):
    """
    Calculates the accuracy of a batch of prediction (defined as the amount of correct predictions in proportion to
    the amount of total predictions).
    :param scores: The output layer of the network, as matrix of size (N x K), where N is the number of observations to
    predict and K is the number of classes.
    :param y: The ground truth labels for the observations as class indices.
    :return: The accuracy value.
    """
    predicted_classes = np.argmax(scores, axis=1)
    correct_classes = len(np.where(predicted_classes == y)[0])
    return correct_classes / len(predicted_classes)


def backpropagation(x, s, y, hidden_layers, wh, bh, w_out, b_out, alpha, lambda_):
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
    :param lambda_: The regularization loss hyperparameter.
    :return: The backpropagation returns relevant gradients as a tuple containing the following values:
    * An array containing the gradients for the connection weights of each hidden layer of the same form as `wh`.
    * An array containing the gradients for the biases of each hidden layer of the same form as `bh`.
    * An array containing the gradients for the connection weights of the output layer of the same form as `w_out`.
    * An array containing the gradients for the biases of the output layer of the same form as `b_out`.
    """
    dscores = cross_entropy_loss_gradient(s, y)
    dw_out = hidden_layers[-1].T.dot(dscores)
    db_out = np.sum(dscores, axis=0, keepdims=True)
    dhiddens = {}
    dwh = [np.full(w_i.shape, .0) for w_i in wh]
    dbh = [np.empty(b_i.shape) for b_i in bh]
    for h in range(len(hidden_layers) - 1, -1, -1):
        if h == len(hidden_layers) - 1:
            dhidden = dscores.dot(w_out.T)
        else:
            dhidden = dhiddens[h + 1].dot(wh[h + 1].T)
        dhidden = activation_gradient(dhidden, hidden_layers[h], alpha)
        dhiddens[h] = dhidden
        if h == 0:
            dwh[h] = x.T.dot(dhidden)
        else:
            dwh[h] = hidden_layers[h - 1].T.dot(dhidden)
        dbh[h] = np.sum(dhidden, axis=0, keepdims=True)
    dw_out += regularization_loss_gradient_by_weights(w_out, lambda_)
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
    x += update
    return x, m, v


def train(x, y, epochs, hidden_sizes, wh, bh, w_out, b_out, learning_rate, p, alpha, beta1, beta2, eps, lambda_,
          batch_size=0):
    """
    Trains a neural network. The learnable parameters `wh`, `bh`, `w_out` and `b_out` are optimized as long as
    `epochs` indicates and then returned.
    :param x: The input data of form (N x D), where N is the number of observations an D is the dimensionality.
    :param y: The ground truth labels for each observation.
    :param epochs: The amount of runs a network should take. (Note: The number of iterations is 2 * N * epochs,
    where N is the number of inputs.)
    :param hidden_sizes: The size of each hidden layer as array.
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
    :param p: The probability of each neuron to be dropped out. Set to 1 to disable dropout.
    :param alpha: The factor by which negative inputs are scaled in ReLU activations. Set to 0 to avoid leaky ReLU.
    :param beta1: Hyperparameter for the Adam parameter update. Recommended to be .9.
    :param beta2: Hyperparameter for the Adam parameter update. Recommended to be .999.
    :param eps: Hyperparameter for the Adam parameter update. Recommended to be 1e-8.
    :param batch_size: The number of input observations to use in each epoch. If this parameter is set to a positive
    value, it will enable Minibatch gradient descent, whereby only a random subset of the input observations is used
    to optimize parameters. This greatly improves performance and yields about the same accuracy as using all input
    observations.
    :return: A tuple containing the optimized learnable parameters `wh`, `bh`, `w_out` and `b_out`.
    """
    m_wh = [.0 for i in range(len(wh))]
    v_wh = [.0 for i in range(len(wh))]
    m_bh = [.0 for i in range(len(bh))]
    v_bh = [.0 for i in range(len(bh))]
    m_w_out = .0
    v_w_out = .0
    m_b_out = .0
    v_b_out = .0
    losses = []
    accuracies = []
    for epoch in range(1, epochs + 1):
        batch_x, batch_y = get_batches(x, y, batch_size)

        # Feed-forward the network
        hidden_layers, outs, w_out = forward_pass(batch_x, hidden_sizes, wh, bh, w_out, b_out, alpha, p)

        # Calculate loss and accuracy measures
        loss = calculate_cross_entropy_loss(outs, batch_y, w_out, lambda_)
        losses.append(loss)
        accuracy = calculate_accuracy(outs, batch_y)
        accuracies.append(accuracy)
        print(epoch, loss, accuracy)

        # Backpropagation
        dwh, dbh, dw_out, db_out = backpropagation(batch_x, outs, batch_y, hidden_layers, wh, bh, w_out, b_out, alpha,
                                                   lambda_)

        # Update parameters using gradients of backpropagation
        for h in range(len(hidden_layers)):
            wh[h], m_wh[h], v_wh[h] =\
                update_parameter(wh[h], dwh[h], epoch, learning_rate, m_wh[h], v_wh[h], beta1, beta2, eps)
            bh[h], m_bh[h], v_bh[h] =\
                update_parameter(bh[h], dbh[h], epoch, learning_rate, m_bh[h], v_bh[h], beta1, beta2, eps)
        w_out, m_w_out, v_w_out =\
            update_parameter(w_out, dw_out, epoch, learning_rate, m_w_out, v_w_out, beta1, beta2, eps)
        b_out, m_b_out, v_b_out =\
            update_parameter(b_out, db_out, epoch, learning_rate, m_b_out, v_b_out, beta1, beta2, eps)
    write_list(losses, 'losses.txt')
    write_list(accuracies, 'accuracies.txt')
    return wh, bh, w_out, b_out


def predict(x, wh, bh, w_out, b_out, alpha=0):
    """
    Predicts the classes of the given input observations.
    :param x: The input observations to classify.
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
    hidden_layers = []
    for h in range(len(wh)):
        if h == 0:
            hidden = x.dot(wh[h]) + bh[h]
        else:
            hidden = hidden_layers[h - 1].dot(wh[h]) + bh[h]
        hidden = calculate_activation(hidden, alpha)
        hidden_layers.append(hidden)
    outs = hidden_layers[-1].dot(w_out) + b_out
    return outs
