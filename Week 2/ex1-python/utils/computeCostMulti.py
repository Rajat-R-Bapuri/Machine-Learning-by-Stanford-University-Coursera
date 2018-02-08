import numpy as np


def computeCostMulti(X_c, y_c, theta_c):
    hypothesis = X_c.dot(theta_c)
    squaredError = np.power((hypothesis[:, 0] - y_c), 2)
    squaredError = squaredError / (2 * X_c.shape[0])
    J = np.sum(squaredError)
    return J
