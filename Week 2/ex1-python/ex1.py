import numpy as np
import matplotlib.pyplot as plt
from utils.warmUpExercise import warmUpExercise
from utils.computeCost import computeCost
from utils.gradientDescent import gradientDescent
from utils.plotData import plotData
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# ==================== Part 1: Basic Function ====================
print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')
print(warmUpExercise())

input('Program paused. Press enter to continue.\n')


# ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
data1 = np.loadtxt("ex1data1.txt", delimiter=',')
X = data1[:, 0]
y = data1[:, 1]
plotData(X, y)
input('Program paused. Press enter to continue.\n')


# =================== Part 3: Cost and Gradient descent ===================
designMatrix = np.column_stack((np.ones(len(X)), X))  # adding a column of ones to x
theta = np.zeros((designMatrix.shape[1], 1))  # parameters that need to be learnt
cost = computeCost(designMatrix, y, theta)  # compute cost when both theta0 and theta1 are zero --> initial cost
print('\nTesting the cost function ...\n')
print('With theta = [0 ; 0]\nCost computed = ', cost, '\n')
print('Expected cost value (approx) 32.07\n')

iterations = 1500
alpha = 0.01

cost = computeCost(designMatrix, y, np.array([[-1], [2]]))
print('\nWith theta = [-1 ; 2]\nCost computed = ', cost, '\n')
print('Expected cost value (approx) 54.24\n')
input('Program paused. Press enter to continue.\n')

print('\nRunning Gradient Descent ...\n')
theta, J = (gradientDescent(designMatrix, y, theta, alpha, iterations))
print('Theta found by gradient descent:\n')
print('\n', theta[0], '\n', theta[1], '\n')
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')

plt.plot(X, designMatrix.dot(theta))

predict1 = np.sum(np.array([1, 3.5]).dot(theta))
print('For population = 35,000, we predict a profit of \n', predict1*10000)
predict2 = np.sum(np.array([1, 7]).dot(theta))
print('For population = 70,000, we predict a profit of ', predict2*10000)


#  ============= Part 4: Visualizing J(theta_0, theta_1) =============
theta0_t = np.linspace(-10, 10, designMatrix.shape[0])
theta1_t = np.linspace(-1, 4, designMatrix.shape[0])
J = np.zeros((X.shape[0], X.shape[0]))

for i in range(theta0_t.size):
    for j in range(theta1_t.size):
        t = np.array([[theta0_t[i]], [theta1_t[j]]])
        J[i, j] = computeCost(designMatrix, y, t)

theta0_t, theta1_t = np.meshgrid(theta0_t, theta1_t)

fig = plt.figure(figsize=(7, 7))
ax = fig.gca(projection='3d')

ax.plot_surface(theta0_t, theta1_t, J, cmap=cm.coolwarm)
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
ax.set_zlabel(r'J($\theta$)')

# Has some problem, contours are shown but a bit disoriented
CS = ax.contour(theta0_t, theta1_t, J, np.logspace(-2, 3, 20))
ax.scatter(theta[0], theta[1], c='r')

plt.show()
