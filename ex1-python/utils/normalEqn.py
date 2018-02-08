from numpy.linalg import inv
import numpy as np


def normalEqn(X_n, y_n):
    theta = np.zeros((X_n.shape[0], 1))
    a = inv(X_n.T.dot(X_n))
    b = a.dot(X_n.T)
    theta = b.dot(y_n)
    return theta
