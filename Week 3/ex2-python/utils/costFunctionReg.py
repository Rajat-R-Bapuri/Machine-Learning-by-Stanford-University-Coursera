import numpy as np
from utils.sigmoid import sigmoid


def costFunctionReg(theta_c, X_c, y_c, lambda_reg):
    m = y_c.shape[0]
    J = 0
    hypothesis = sigmoid(X_c.dot(theta_c))
    J = (1 / m) * np.sum((-y_c.T.dot(np.log(hypothesis))) - ((1 - y_c).T.dot(np.log(1 - hypothesis)))) + \
        (lambda_reg / (2 * m)) * np.sum(np.power(theta_c[1:], 2))
    return J


# The optimization function (scipy.optimize.fmin) in scipy needs only cost to be returned,
# so another function has been written to return only cost
def gradientReg(theta_c, X_c, y_c, lambda_reg):
    m = y_c.shape[0]
    grad = np.zeros((theta_c.shape[0], 1))
    grad = (1 / m) * (X_c.T.dot(sigmoid(X_c.dot(theta_c)) - y_c))
    grad[1:] = grad[1:] + theta_c[1:] * lambda_reg / m
    return grad
