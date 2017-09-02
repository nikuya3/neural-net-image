from csv import reader
from net import initialize_parameters, train, predict
import numpy as np
from pickle import dump, load


def split_data(x, y, tr):
    training_size = int(len(x) * tr)
    indices = np.random.permutation(x.shape[0])
    training_idx, test_idx = indices[:training_size], indices[training_size:]
    x_tr, x_te = x[training_idx, :], x[test_idx, :]
    y_tr, y_te = y[training_idx], y[test_idx]
    return x_tr, x_te, y_tr, y_te


def get_iris_data():
    x = []
    y = []
    prev_y = None
    y_counter = 0
    with open('data/iris.data', 'r') as file:
        iris = list(reader(file))
        for row_index in range(len(iris)):
            row = iris[row_index]
            for column_index in range(len(row)):
                column = row[column_index]
                if column_index == len(row) - 1:
                    if prev_y:
                        if column != prev_y:
                            y_counter += 1
                        val = y_counter
                    else:
                        val = 0
                    prev_y = column
                    y.append(val)
                else:
                    val = float(column)
                    if row_index < len(x):
                        x[row_index].append(val)
                    else:
                        x.append([val])
    return x, y


x, y = get_iris_data()
x = np.array(x)
y = np.array(y)
x_tr, x_te, y_tr, y_te = split_data(x, y, .8)

alpha = .0
beta1 = .9
beta2 = .999
eps = 1e-8
lambda_ = 0
epochs = 100
learning_rate = .0005
p = .9
hidden_sizes = [1000, 100]
out_size = np.unique(y_tr).shape[0]

wh, bh, w_out, b_out = initialize_parameters(x_tr[0].shape[0], hidden_sizes, out_size)
wh, bh, w_out, b_out = train(x_tr, y_tr, epochs, hidden_sizes, wh, bh, w_out, b_out, learning_rate, p, alpha, beta1,
                             beta2, eps, lambda_)

with open('dump-iris.p', 'wb') as dump_file:
    dump((wh, w_out, bh, b_out), dump_file)

with open('dump-iris.p', 'rb') as file:
    wh, wo, bh, bo = load(file)
    predicted_class_scores = predict(x_te, wh, bh, wo, bo, alpha)
    predicted_classes = np.argmax(predicted_class_scores, axis=1)
    correct_classes = len(np.where(predicted_classes == y_te)[0])
    print('Test accuracy of iris network:', correct_classes / len(x_te))
    fixed_data = np.array([[4.7, 3.2, 1.3, .2], [6.6, 2.9, 4.6, 1.3], [5.8, 2.8, 5.1, 2.4]])
    print('Sample output:', predict(fixed_data, wh, bh, wo, bo, alpha))