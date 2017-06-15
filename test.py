from keras.models import Sequential
from keras.layers import Dense, Activation, GaussianDropout
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam
from keras.regularizers import l2
from pickle import load
import numpy as np


def unpickle(file):
    with open(file, 'rb') as fo:
        data = load(fo, encoding='bytes')
    return data


def get_data():
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

# Hyperparameters
alpha = .0  # Slope for leaky ReLU. Set to 0 to avoid leaky ReLU.
beta1 = .9  # Hyperparameter for Adam parameter update
beta2 = .999  # Hyperparameter for Adam parameter update
delta = 1  # The minimum margin of the hinge loss
eps = 1e-8  # Hyperparameter for Adam parameter update
lambda_ = .1  # The regularization strength (has an influence on regularization loss).
epochs = 50  # The amount of 'iterations' the network should take
learning_rate = .001  # The step size for each epoch (influences how greedy the network changes its parameters)
p = .75  # Dropout rate as the possibility of each neuron to be dropped out.

x_tr, y_tr = get_data()

k = len(np.unique(y_tr))  # number of classes
hidden_sizes = [1000, 100]

y_tr = to_categorical(y_tr, 10)

model = Sequential()

model.add(Dense(units=1000, input_shape=x_tr[0].shape, activation='relu'))
model.add(GaussianDropout(p))
model.add(Dense(units=100, activation='relu'))
model.add(GaussianDropout(p))
model.add(Dense(k, activation='softmax', kernel_regularizer=l2(.1)))

adam = Adam(lr=learning_rate, beta_1=beta1, beta_2=beta2, epsilon=eps, decay=.0)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

model.fit(x_tr, y_tr, epochs=100)
loss_and_metrics = model.evaluate(x_tr, y_tr)
print(loss_and_metrics)