from classes import *
from functions import * 


import matplotlib.pyplot as plt
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
train = x_train[0:1000].reshape(1000, -1) / 255 
labels = y_train[0:1000]
temp = np.zeros((1000, 10))
for i, l in enumerate(labels):
    temp[i][l] = 1
labels = temp
iterations = 1000
lr = 1e-2


model = Model([
    Dense((784,256)),
    ReLU(),
    Dense((256, 256)),
    ReLU(),
    Dense((256, 256)),
    ReLU(),
    Dense((256, 256)),
    ReLU(),
    Dense((256,10))
])

error = []
for i in range(iterations):
    pred = model.forward(train)
    grad = MSE(labels, pred, deriv=True)

    error.append(MSE(labels, pred)) 
    print(f'{i}. Error: {error[-1]}')

    model.backward(grad,lr)