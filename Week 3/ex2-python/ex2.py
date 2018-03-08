import numpy as np
import scipy.optimize as op
from utils.plotData import plotData
from utils.costFunction import costFunction,gradient
from utils.plotDecisionBoundary import plotDecisionBoundary
from utils.sigmoid import sigmoid
from utils.predict import predict

data = np.loadtxt('ex2data1.txt', delimiter=',')
X = np.array(data[:, :2])
y = np.reshape(data[:, 2], (X.shape[0], 1))


# ==================== Part 1: Plotting ====================
print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')
plotData(X, y)

input('\nProgram paused. Press enter to continue.\n')

# ============ Part 2: Compute Cost and Gradient ============
designMatrix = np.column_stack((np.ones(len(X)), X))  # Add intercept term to x and X_test
initial_theta = np.zeros((designMatrix.shape[1], 1))  # Initialize fitting parameters
cost = costFunction(initial_theta, designMatrix, y)
grad = gradient(initial_theta, designMatrix, y)
print('Cost at initial theta (zeros):\n', cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros):\n')
print(grad[0], '\n', grad[1], '\n', grad[2], '\n')
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([[-24], [0.2], [0.2]])
cost = costFunction(test_theta, designMatrix, y)
grad = gradient(test_theta, designMatrix, y)
print('\nCost at test theta: \n', cost)
print('Expected cost (approx): 0.218\n')
print('Gradient at test theta: \n')
print(grad[0], '\n', grad[1], '\n', grad[2], '\n')
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

input('\nProgram paused. Press enter to continue.\n')


# ============= Part 3: Optimizing using optimize  =============
# Run scipy.optimize.fmin to obtain the optimal theta
# This function will return theta and the cost
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin.html
optimizedTheta = op.fmin(costFunction, x0=initial_theta, args=(designMatrix, y), maxiter=400)

# Print theta to screen
print('Cost at theta found by fmin: \n',  costFunction(optimizedTheta, designMatrix, y))
print('Expected cost (approx): 0.203\n')
print('Theta: \n')
print(optimizedTheta[0], '\n', optimizedTheta[1], '\n', optimizedTheta[2], '\n')
print('Expected theta (approx):\n')
print(' -25.161\n 0.206\n 0.201\n')

# Plot Boundary
plotDecisionBoundary(optimizedTheta, designMatrix, y)

input('\nProgram paused. Press enter to continue.\n')


# ============== Part 4: Predict and Accuracies ==============
testx = np.array([1, 45, 85])
prob = sigmoid(testx.dot(optimizedTheta))
print(['For a student with scores 45 and 85, we predict an admission probability of \n'], prob)
print('Expected value: 0.775 +/- 0.002\n\n')

# Compute accuracy on our training set
p = predict(optimizedTheta, designMatrix)

# element wise comparison of predictions made with the training data and calculating the accuracy %
print('Train Accuracy:', np.mean((p == y)) * 100)
print('Expected accuracy (approx): 89.0\n')
