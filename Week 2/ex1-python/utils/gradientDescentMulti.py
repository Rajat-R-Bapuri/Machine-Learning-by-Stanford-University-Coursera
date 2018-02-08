import numpy as np
from utils.computeCostMulti import computeCostMulti


def gradientDescentMulti(X_g, y_g, theta_g, alpha, iterations):
    m = X_g.shape[0]
    J_history = np.zeros((iterations,1))
    for i in range(iterations):
        temp = theta_g.copy()
        for j in range(theta_g.shape[0]):
            delta = X_g.dot(theta_g) - y_g
            temp[j] = temp[j] - (alpha * np.sum(delta.T * X_g[:, j]) / m)
        theta_g = temp
        J_history[i] = computeCostMulti(X_g, y_g, theta_g)
    return theta_g, J_history
