from classes import *
from functions import * 


import matplotlib.pyplot as plt
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

train = x_train.reshape(-1, 784) / 255 
test = x_test.reshape(-1, 784) / 255


def ohe(labels):
    temp = np.zeros((len(labels), 10))
    for i,l in enumerate(labels):
        temp[i][l] = 1
    return temp


train_labels = ohe(y_train)
test_labels = ohe(y_test)


model = Model([
    Dense((784,8)),
    ReLU(),
    Dense((8, 10))
])

model.train(train, train_labels, lr=0.001, batch_size=32, loss=MSE, epochs=100, val_data=(test, test_labels))