from numpy.random import random
from numpy import array, sqrt, outer

# THE PARENT CLASS, FROM WHICH ALL OTHER LAYERS DERIVE #
class Layer:
    def __init__(self, previous_layer=None, name='parent_layer'):
        self.name = name
        self.previous_layer = previous_layer
        if self.previous_layer != None:
            self.previous_layer.next_layer = self 
        self.next_layer = None

# THE DENSE LAYER CLASS #
class Dense(Layer):
    def __init__(self, input_shape, output_shape, name='dense_layer', previous_layer=None):
        super().__init__(previous_layer, name)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = random((self.input_shape, self.output_shape))
        self.bias = random((1, self.output_shape))

    def forward(self, input):
        self.input = input
        self.output = self.input.dot(self.weights)
        if self.next_layer == None:
            return self.output
        self.next_layer.forward(self.output)

    def backward(self, gradient, lr):
        weight_grad = outer(self.input, gradient)
        bias_grad = gradient 
        self.weights -= lr * weight_grad
        self.bias -= lr * bias_grad

        input_grad = gradient.dot(self.weights.T)

        if self.previous_layer != None:
            self.previous_layer.backward(input_grad)