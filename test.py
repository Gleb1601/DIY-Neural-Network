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
    Dense((784,16)),
    ReLU(),
    Dense((16, 10))
])

hist = model.train(train, train_labels, lr=0.001, batch_size=32, loss=mse, metric=acc_score, epochs=10, val_data=(test, test_labels))
print(hist)


train_loss  = hist['train_loss']
plt.plot(train_loss)
plt.show()

