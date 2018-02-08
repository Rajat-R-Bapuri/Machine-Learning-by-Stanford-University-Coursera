import numpy as np
from utils.computeCost import computeCost


def gradientDescent(X, y, theta, alpha, num_iters):
    J_history = np.zeros((num_iters, 1))
    m = len(y)
    for i in range(num_iters):
        delta = X.dot(theta)[:, 0] - y
        temp1 = theta[0] - (alpha * np.sum(delta.T * X[:, 0])/m)
        temp2 = theta[1] - (alpha * np.sum(delta.T * X[:, 1])/m)
        theta[0] = temp1
        theta[1] = temp2
        J_history[i] = computeCost(X, y, theta)

    return theta, J_history