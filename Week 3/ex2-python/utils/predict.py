import numpy as np
from utils.sigmoid import sigmoid


def predict(theta_p, X_p):
    m = X_p.shape[0]  # Number of training examples
    p = np.zeros((m, 1))
    h = sigmoid(X_p.dot(theta_p))
    for i in range(m):
        if h[i] >= 0.5:
            p[i] = 1
        else:
            p[i] = 0

    return p