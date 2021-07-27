# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
#from matplotlib import pyplot
#from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces

# library written for this exercise providing additional functions for assignment submission, and others
#import utils

# Read comma separated data
data = np.loadtxt(os.path.join('Data', 'ex1data1.txt'), delimiter=',')
X, y = data[:, 0], data[:, 1]

m = y.size  # number of training examples

# Add a column of ones to X. The numpy function stack joins arrays along a given axis.
# The first axis (axis=0) refers to rows (training examples)
# and second axis (axis=1) refers to columns (features).
X = np.stack([np.ones(m), X], axis=1)


def hypothesis(theta, X):
    return theta[0] + theta[1] * X[:, 1]


def computeCost(X, y, theta):
    # initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly
    J = 0

    J = (1 / (2 * m)) * np.sum((hypothesis(theta, X) - y) ** 2)

    # ====================== YOUR CODE HERE =====================

    # ===========================================================
    return J


#J = computeCost(X, y, theta=np.array([0.0, 0.0]))
#print('With theta = [0, 0] \nCost computed = %.2f' % J)
#print('Expected cost value (approximately) 32.07\n')

# further testing of the cost function
#J = computeCost(X, y, theta=np.array([-1, 2]))
#print('With theta = [-1, 2]\nCost computed = %.2f' % J)
#print('Expected cost value (approximately) 54.24')


def gradientDescent(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn `theta`. Updates theta by taking `num_iters`
    gradient steps with learning rate `alpha`.
    Parameters
    ----------
    X : array_like
        The input dataset of shape (m x n+1).
    y : array_like
        Value at given features. A vector of shape (m, ).
    theta : array_like
        Initial values for the linear regression parameters.
        A vector of shape (n+1, ).
    alpha : float
        The learning rate.
    num_iters : int
        The number of iterations for gradient descent.
    Returns
    -------
    theta : array_like
        The learned linear regression parameters. A vector of shape (n+1, ).
    J_history : list
        A python list for the values of the cost function after each iteration.
    Instructions
    ------------
    Peform a single gradient step on the parameter vector theta.
    While debugging, it can be useful to print out the values of
    the cost function (computeCost) and gradient here.
    """
    # Initialize some useful values
    m = y.shape[0]  # number of training examples

    # make a copy of theta, to avoid changing the original array, since numpy arrays
    # are passed by reference to functions
    theta = theta.copy()
    theta0 = []
    theta1 = []
    J_history = []  # Use a python list to save cost in every iteration

    X_Values = X[:, 1]
    for i in range(num_iters):
        # ==================== YOUR CODE HERE =================================
        h = hypothesis(theta, X)
        theta[0] = theta[0] - (alpha/m)*(np.sum(h-y))
        theta[1] = theta[1] - (alpha/m) * (np.dot((h - y), X[:, 1]))
        # ===================================================================
        # save the cost J in every iteration
        J_history.append(computeCost(X, y, theta))
        # print(J_history)
    return theta, J_history


# initialize fitting parameters
theta = np.zeros(2)

# some gradient descent settings
iterations = 1500
alpha = 0.01

theta, J_history, theta1 = gradientDescent(X, y, theta, alpha, iterations)
print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(*theta))
print('Expected theta values (approximately): [-3.6303, 1.1664]')