import numpy as np
import matplotlib.pyplot as plt


def plotData(X, y):
    fig, ax = plt.subplots()
    positives = np.where(y == 1)  # get indices for the positives
    negatives = np.where(y == 0)  # get indices for the negatives
    ax.plot(X[positives[0], 0], X[positives[0], 1], 'bo', marker='+', label='Admitted')
    ax.plot(X[negatives[0], 0], X[negatives[0], 1], 'ro', marker='o', label='Not Admitted')
    ax.set_xlabel('Exam 1 score')
    ax.set_ylabel('Exam 2 score')
    ax.legend()
    fig