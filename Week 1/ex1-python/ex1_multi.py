import numpy as np
import matplotlib.pyplot as plt
from utils.featureNormalize import featureNormalize
from utils.computeCostMulti import computeCostMulti
from utils.gradientDescentMulti import gradientDescentMulti
from utils.normalEqn import normalEqn
import itertools
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# ================ Part 1: Feature Normalization ================
print('Loading data ...\n')
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, :2]
y = np.reshape(data[:, 2], (X.shape[0], 1))
m = X.shape[0]
X = np.column_stack((np.ones(len(X)), X))

# Print out some data points
print('First 10 examples from the dataset: \n')
print(' x =', X[:10, :], '\ny =',  y[:10], '\n')

input('Program paused. Press enter to continue.\n')

# Scale features and set them to zero mean
print('Normalizing Features ...\n')
normalizedX, mu, sigma = featureNormalize(X)

# ================ Part 2: Gradient Descent ================
print('Running gradient descent ...\n')

alpha = 0.03
num_iters = 1500

# Init Theta and Run Gradient Descent
theta = np.zeros((normalizedX.shape[1], 1))
theta, J_history = gradientDescentMulti(normalizedX, y, theta, alpha, num_iters)

# Plot the convergence graph
plt.figure(figsize=(10, 6))
plt.plot(range(len(J_history)), J_history, 'bo')
plt.grid(True)
plt.title("Convergence graph")
plt.xlabel("Number of iterations")
plt.ylabel("Cost J")
plt.show()

# Display gradient descent's result
print('Theta computed from gradient descent: \n')
print('theta0=', theta[0], '\ntheta1=', theta[1], '\ntheta2=', theta[2], '\n')

# Estimate the price of a 1650 sq-ft, 3 br house
testx = np.array([1., 1650., 3.])
testx[1] = (testx[1] - mu[1])/sigma[1]
testx[2] = (testx[2] - mu[2])/sigma[2]
predict_g = np.dot(testx,theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):', predict_g[0], '\n')

input('Program paused. Press enter to continue.\n')


# ================ Part 3: Normal Equations ================
print('Solving with normal equations...\n')
theta = normalEqn(X, y)

# Display normal equation's result
print('Theta computed from the normal equations: \n')
print('theta0=', theta[0], '\ntheta1=', theta[1], '\ntheta2=', theta[2], '\n')

# Estimate the price of a 1650 sq-ft, 3 br house
predict_n = np.array([1., 1650., 3.]).dot(theta)
print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):', predict_n[0])

