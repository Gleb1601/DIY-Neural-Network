import numpy as np

def mse(true, pred, deriv=False):
    if deriv:
        return (pred - true) / len(pred)
    return np.mean((pred - true) ** 2) / 2 


def acc_score(true, pred):
    return np.mean(np.argmax(true, axis=1) == np.argmax(pred, axis=1)) 