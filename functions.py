import numpy as np

def MSE(true, pred, deriv=False):
    if deriv:
        return (pred - true) / len(pred)
    return np.mean((pred - true) ** 2) / 2 