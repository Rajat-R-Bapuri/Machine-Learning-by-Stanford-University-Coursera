import numpy as np
import matplotlib.pyplot as plt
from utils.plotDataReg import plotDataReg
from utils.mapFeature import mapFeature

def plotDecisionBoundaryReg(theta_p, X_p, y_p):
    plotDataReg(X_p[:, 1:], y_p)
    if X_p.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([np.min(X_p[:, 1]) - 2., np.max(X_p[:, 1]) + 2.])

        # Calculate the decision boundary line
        plot_y = (-1. / theta_p[2]) * (theta_p[0] + theta_p[1] * plot_x)

        plt.plot(plot_x, plot_y, label='Decision Boundary')
        plt.legend()
        plt.show()

    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        z = np.zeros((len(u), len(v)))
        # Evaluate z = theta*x over the grid
        for i in range(len(u)):
            for j in range(len(v)):
                z[i][j] = mapFeature(np.array([u[i]]), np.array([v[j]])).dot(theta_p)

        z = z.T  # important to transpose z before calling contour

        u, v = np.meshgrid(u, v)
        plt.contour(u, v, z, [0])
        plt.show()
