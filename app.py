from argparse import ArgumentParser
from net import predict
import numpy as np
from pickle import load
from scipy.ndimage import imread
from scipy.misc import imresize
from os import listdir
from os.path import isfile, join

filenames = [join('img', f) for f in listdir('img') if isfile(join('img', f))]
# parser = ArgumentParser(description='Detects objects in an image', prog='app')
# parser.add_argument('file', type=str, help='The path to the image')
# args = parser.parse_args()
# filename = args.file
class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
with open('data/batches.meta', 'rb') as fo:
    data = load(fo, encoding='bytes')

with open('dump_best4.p', 'rb') as file:
    _, _, _, _, pre_mean, pre_std = load(file)

images = []
for name in filenames:
    print()
    print('--------------------------------------------')
    print()
    print(name)
    image = imread(name)
    image = imresize(image, (32, 32))
    image = image.flatten()
    image = image.astype(float)
    image -= pre_mean
    image /= pre_std
    images.append(image)

with open('dump_best5.p', 'rb') as file:
    wh, wo, bh, bo = load(file)
    class_scores = predict(np.array(images), wh, bh, wo, bo)
    classes = np.argmax(class_scores, axis=1)
    for i in range(len(classes)):
        print(filenames[i], class_labels[classes[i]], class_scores[i, classes[i]])
    # label_scores = {}
    # for nr in range(len(class_scores[0])):
    #     label_scores[class_labels[nr]] = class_scores[0][nr]
    # for k in sorted(label_scores, key=label_scores.get, reverse=True):
    #     print(k, label_scores[k])
