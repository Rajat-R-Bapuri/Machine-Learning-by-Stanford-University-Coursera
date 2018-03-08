import numpy as np


def sigmoid(z):
    return np.reciprocal(1 + np.exp(-z))
