import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.regularizers import l2
from net import initialize_parameters, train, predict
import numpy as np
from pickle import dump, load


def unpickle(file):
    with open(file, 'rb') as fo:
        data = load(fo, encoding='bytes')
    return data


def get_training_data():
    """
    Reads input data from the 'data' directory.
    :return: A matrix of form 50000 x 3072
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


def get_test_data():
    """
    Reads test data from the 'data' directory.
    :return: A matrix of form 10000 x 3072
    """
    batch = unpickle('data/test_batch')
    return batch[b'data'], batch[b'labels']


def preprocess_data(train, test, validation=None):
    """
    Applies zero-centering and normalization of input data for better model performance.
    :param train: A numpy matrix containing the data the network is trained on.
    :param test: A numpy matrix containing the data used to test the network.
    :param validation: An optional numpy matrix containing the data used to apply hyperparameter validation on the
    network.
    :return: The preprocessed matrices.
    """
    train = train.astype(float)
    if validation is not None:
        validation = validation.astype(float)
    test = test.astype(float)
    # Zero-centering (subtracting the mean)
    mean = np.mean(train, axis=0)  # Using statistic of training set
    train -= mean
    if validation is not None:
        validation -= mean
    test -= mean
    # Normalization of data dimension to be of equal scale (division by standard deviation)
    std = np.std(train, axis=0)
    train /= std
    if validation is not None:
        validation /= std
    test /= std
    return train, validation, test, mean, std


def accuracy(x, y, wh, wo, bh, bo, alpha):
    """
    Measures the accuracy of a nerual network. Specifically the proportion of correct predictions of input data x
    using the parameters wh, wo, bh, bo and the ground truth labels y.
    :param x: The input data to be predicted.
    :param y: The ground truth labels for the input data.
    :param wh: The weights of each hidden layer as an array.
    :param wo: The weights of the output layer.
    :param bh: The biases of each hidden layer as an array.
    :param bo: The biases of the output layer.
    :param alpha: The factor by which negative inputs are scaled in ReLU activations. Set to 0 to avoid leaky ReLU.
    :return: The accuracy as proportion, where 1 indicates a perfect match and 0 indicates a perfect mismatch.
    """
    predicted_class_scores = predict(x, wh, bh, wo, bo, alpha)
    predicted_classes = np.argmax(predicted_class_scores, axis=1)
    correct_classes = len(np.where(predicted_classes == y)[0])
    return correct_classes / len(x)


def test_neural_net():
    # Parameter initialization
    wh, bh, w_out, b_out = initialize_parameters(x_tr[0].shape[0], hidden_sizes, out_size)

    # Train the network
    train(x_tr, y_tr, epochs, hidden_sizes, wh, bh, w_out, b_out, learning_rate, p, alpha, beta1, beta2, eps, lambda_,
          batch_size)

    # Save parameters for reuse
    with open('dump.p', 'wb') as dump_file:
        dump((wh, w_out, bh, b_out), dump_file)

    # Quick accuracy
    with open('dump.p', 'rb') as file:
        wh, wo, bh, bo = load(file)
        print('Test accuracy of network 1:', accuracy(x_te, y_te, wh, wo, bh, bo, alpha))


def test_keras_net():
    y_tr_cat = to_categorical(y_tr, 10)
    y_te_cat = to_categorical(y_te, 10)

    model = Sequential()
    model.add(Dense(units=hidden_sizes[0], input_shape=x_tr[0].shape, activation='relu'))
    #model.add(Dropout(p))
    for size in hidden_sizes[1:]:
        model.add(Dense(units=size, activation='relu'))
        #model.add(Dropout(p))
    model.add(Dense(out_size, kernel_regularizer=l2(lambda_)))

    adam = Adam(lr=learning_rate, beta_1=beta1, beta_2=beta2, epsilon=eps, decay=.0)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    history = model.fit(x_tr, y_tr_cat, epochs=30, batch_size=256)

    loss_and_metrics = model.evaluate(x_te, y_te_cat, batch_size=256)
    plt.plot(history.history['loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    plt.plot(history.history['acc'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
    print('Test accuracy of network 2:', loss_and_metrics[1])

# hyperparameters
alpha = .0  # Slope for leaky ReLU. Set to 0 to avoid leaky ReLU.
beta1 = .9  # Hyperparameter for Adam parameter update.
beta2 = .999  # Hyperparameter for Adam parameter update.
delta = 1  # The minimum margin of the hinge loss.
eps = 1e-8  # Hyperparameter for Adam parameter update.
lambda_ = 0  # The regularization strength (has an influence on regularization loss).
batch_size = 512  # The size of batches (used to speed up training).
epochs = 200  # The amount of 'iterations' the network should take
learning_rate = .00005  # The step size for each epoch (influences how greedy the network changes its parameters).
p = .75  # Dropout rate as the possibility of each neuron to be dropped out.

# Input data: 50000 training observations, 10000 test observations (no validation set).
x_tr, y_tr = get_training_data()
x_te, y_te = get_test_data()

# x_tr, x_val, x_te = split_data(x_data, .8, .1, .1)
# y_tr, y_val, y_te = split_data(y_data, .8, .1, .1)

# Preprocess data
x_tr, _, x_te, pre_mean, pre_std = preprocess_data(x_tr, x_te)
# wh, wo, bh, bo = unpickle('dump.p')
# with open('dump.p', 'wb') as file:
#     dump((wh, wo, bh, bo, pre_mean, pre_std), file)

# Neural net: IN (3072 x 1) -> HL (1000 x 1) -> HL (250 x 1) -> HL (100 x 1) -> OUT (10 x 1)
hidden_sizes = [4000, 1000, 4000]
out_size = np.unique(y_tr).shape[0]  # number of classes

test_neural_net()
test_keras_net()
