import numpy as np
import functions as fs 

# DENSE LAYER CLASS #
class Dense:

    def info(self):
        print(f'weights_shape: {self.weights.shape}')
        print(f'bias_shape: {self.bias.shape}')


    def __init__(self, shape):
        self.shape = shape
        self.prev = None
        self.next = None
        self.weights = np.random.random(self.shape) * 0.2 - 0.1
        self.bias = np.random.random((1, self.shape[1])) * 0.2 - 0.1


    def forward(self, input):
        self.input = input
        assert(input.shape[1] == self.weights.shape[0])

        output = input.dot(self.weights) + self.bias

        if self.next == None:
            return output
        
        return self.next.forward(output)


    def backward(self, gradient, lr, return_grads=False):

        weights_grad = self.input.T.dot(gradient)
        input_grad = gradient.dot(self.weights.T)
        bias_grad = np.sum(gradient, axis=0, keepdims=True)

        self.weights -= lr * weights_grad
        self.bias -= lr * bias_grad

        if self.prev != None:
            self.prev.backward(input_grad, lr)

        if return_grads:
            return (weights_grad, bias_grad, input_grad)

            
# RELU ACTIVATION CLASS # 
class ReLU:
    def __init__(self):
        self.prev = None
        self.next = None 

    def forward(self, input):
        self.input = input
        return self.next.forward( self.input * (self.input >= 0) )

    def backward(self, gradient, lr):
        grad = gradient * (self.input >= 0)
        self.prev.backward(grad, lr)



# MODEL CLASS, THAT COMBINES SEVERAL LAYERS #
class Model:
    def __init__(self, layers):
        self.layers = layers

        for i in range(len(self.layers)-1, 0, -1):
            self.layers[i].prev = self.layers[i-1]

        for i in range(0, len(self.layers)-1):
            self.layers[i].next = self.layers[i+1]

    def forward(self, input):
        return self.layers[0].forward(input)
    def backward(self, grad, lr):
        self.layers[-1].backward(grad, lr)

    def train(self, X, y, lr, batch_size, loss, metric,  epochs, val_data=None):
        (n_samples, n_features)  = X.shape
        history = {'train_loss' : [], 'train_metric' : [], 'test_loss' : [], 'test_metric' : []}
        for epoch in range(epochs):

            epoch_error = 0.0
            epoch_metric = 0.0
            n_seen = 0 
            test_error = None
            test_acc = None

            for batch_start in range(0, n_samples, batch_size):

                batch_end = min(batch_start+batch_size, n_samples)

                input = X[batch_start:batch_end]
                labels = y[batch_start:batch_end]

                pred = self.forward(input)

                batch_error = loss(labels, pred)
                gradient = loss(labels, pred, deriv=True)
                batch_metric = metric(labels, pred)

                self.backward(gradient, lr)

                epoch_error += batch_error * len(input)
                epoch_metric += batch_metric * len(input)
                n_seen += len(input)

            epoch_metric /= n_seen
            epoch_error /= n_seen

            X_test, y_test = None, None
            if val_data != None:

                X_test, y_test = val_data
                pred = self.forward(X_test)
                test_error = loss(y_test, pred)
                test_metric = metric(y_test, pred)

            history['train_loss'].append(epoch_error)
            history['train_metric'].append(epoch_metric)
            history['test_loss'].append(test_error)
            history['test_metric'].append(test_metric)

            print(f'epoch {epoch}: train_loss: {epoch_error}, train_acc: {epoch_metric}, test_error: {test_error}, test_acc: {test_metric}')

        return history
