import numpy as np

def featureNormalize(X_n):
    tempx = X_n.copy()
    mu = np.mean(tempx, axis=0)
    sigma = np.std(tempx, axis=0)
    tempx[:, 1] = np.divide(np.subtract(tempx[:, 1], mu[1]), sigma[1])
    tempx[:, 2] = np.divide(np.subtract(tempx[:, 2], mu[2]), sigma[2])
    return tempx, mu, sigma