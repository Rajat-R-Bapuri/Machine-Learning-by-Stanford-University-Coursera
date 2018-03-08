import numpy as np
import matplotlib.pyplot as plt


def plotDataReg(X, y):
    fig, ax = plt.subplots()
    positives = np.where(y == 1)  # get indices for the positives
    negatives = np.where(y == 0)  # get indices for the negatives
    ax.plot(X[positives[0], 0], X[positives[0], 1], 'bo', marker='+', label='y = 1')
    ax.plot(X[negatives[0], 0], X[negatives[0], 1], 'ro', marker='o', label='y = 0')
    ax.set_xlabel('Microchip test 1')
    ax.set_ylabel('Microchip test 1')
    ax.legend()
    fig