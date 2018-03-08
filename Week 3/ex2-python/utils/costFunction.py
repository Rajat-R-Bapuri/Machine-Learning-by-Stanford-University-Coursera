import numpy as np
from utils.sigmoid import sigmoid


def costFunction(theta_c, X_c, y_c):
    m = y_c.shape[0]
    J = 0
    hypothesis = sigmoid(X_c.dot(theta_c))
    J = (1 / m) * np.sum((-y_c.T.dot(np.log(hypothesis))) - ((1 - y_c).T.dot(np.log(1 - hypothesis))))
    return J

# The optimization function (scipy.optimize.fmin) in scipy needs only cost to be returned,
# so another function has been written to return only cost


def gradient(theta_c, X_c, y_c):
    return (1 / y_c.shape[0]) * (X_c.T.dot(sigmoid(X_c.dot(theta_c)) - y_c))
