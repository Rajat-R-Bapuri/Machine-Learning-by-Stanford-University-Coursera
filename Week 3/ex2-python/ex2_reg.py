import numpy as np
import scipy.optimize as op
from utils.plotData import plotData
from utils.mapFeature import mapFeature
from utils.costFunctionReg import costFunctionReg,gradientReg
from utils.plotDecisionBoundary import plotDecisionBoundary
from utils.predict import predict


# Load Data
# The first two columns contains the X values and the third column contains the label (y).
data = np.loadtxt('ex2data2.txt', delimiter=',')
X = np.array(data[:, :2])
y = np.reshape(data[:, 2], (X.shape[0], 1))

plotData(X, y)

# =========== Part 1: Regularized Logistic Regression ============
designMatrix = np.column_stack((np.ones(len(X)), X))  # Add intercept term to x and X_test
X = mapFeature(designMatrix[:, 1], designMatrix[:, 2])
initial_theta = np.zeros((X.shape[1], 1))  # Initialize fitting parameters
lambda_reg = 1

cost = costFunctionReg(initial_theta, X, y, lambda_reg)
grad = gradientReg(initial_theta, X, y, lambda_reg)
print('Cost at initial theta (zeros): ', cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros) - first five values only:\n')
print('\n', grad[:5])
print('Expected gradients (approx) - first five values only:\n')
print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')

# Compute and display cost and gradient
# with all-ones theta and lambda = 10
test_theta = np.ones((X.shape[1], 1))
cost = costFunctionReg(test_theta, X, y, 10)
grad = gradientReg(test_theta, X, y, 10)

print('\nCost at test theta (with lambda = 10): \n', cost)
print('Expected cost (approx): 3.16\n')
print('Gradient at test theta - first five values only:\n')
print('\n', grad[:5])
print('Expected gradients (approx) - first five values only:\n')
print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n')

input('\nProgram paused. Press enter to continue.\n')


# ============= Part 2: Regularization and Accuracies =============
initial_theta = np.zeros((X.shape[1], 1))  # Initialize fitting parameters
lambda_reg = 1
optimizedTheta = op.fmin_bfgs(costFunctionReg, x0=initial_theta, args=(X, y, lambda_reg), maxiter=1000)
plotDecisionBoundary(optimizedTheta, X, y)

# Compute accuracy on our training set
p = predict(optimizedTheta, X)

# element wise comparison of predictions made with the training data and calculating the accuracy %
print('Train Accuracy:', np.mean((p == y)) * 100)
print('Expected accuracy (approx): 89.0\n')
