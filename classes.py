import numpy as np

# THE DENSE LAYER CLASS #
class Dense:
    def __init__(self, weights_shape):
        self.input_shape = weights_shape[0]
        self.output_shape = weights_shape[1]
        self.weights = np.random.random((self.input_shape, self.output_shape))
        self.bias = np.random.random((1, self.output_shape))

    def forward(self, input):
        self.input = input
        self.output = self.input.dot(self.weights) + self.bias
        return self.output

    def backward(self, gradient, lr):

        input_grad = gradient.dot(self.weights.T)
        weight_grad = self.input.T.dot(gradient)
        bias_grad = np.sum(gradient, axis=0, keepdims=True) / len(y)

        self.weights -= lr * weight_grad
        self.bias -= lr * bias_grad

        return input_grad

