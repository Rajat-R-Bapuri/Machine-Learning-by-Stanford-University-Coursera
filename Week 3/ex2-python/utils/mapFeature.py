import numpy as np


def mapFeature(x1, x2):
    degree = 6
    out = np.ones((x1.shape[0], 1))
    for i in range(1, degree + 1):
        for j in range(i+1):
            out = np.column_stack((out, (np.power(x1, (i - j))) * (np.power(x2, j))))
    return out
