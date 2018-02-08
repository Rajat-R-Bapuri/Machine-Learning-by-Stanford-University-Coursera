import numpy as np


def computeCost(X, y, theta):
    hypothesis = X.dot(theta)
    squaredError = np.power(hypothesis[:, 0] - y, 2)
    squaredError = squaredError/(2*len(X))
    J = np.sum(squaredError)
    return J