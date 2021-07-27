import os

# Scientific and vector computation for python
import matplotlib
import numpy as np

# Plotting library
from jedi.api.refactoring import inline
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D  # needed to plot 3-D surfaces

data = np.loadtxt(os.path.join('Data', 'ex1data2.txt'), delimiter=',')
X, y = data[:, :2], data[:, 2]

m = y.size  # number of training examples
# X = np.stack([np.ones(m), X], axis=1)

fig = pyplot.figure()  # open a new figure


# print out some data points
# print('{:>8s}{:>8s}{:>10s}'.format('X[:,0]', 'X[:, 1]', 'y'))
# print('-' * 26)
# for i in range(10):
#    print('{:8.0f}{:8.0f}{:10.0f}'.format(X[i, 0], X[i, 1], y[i]))


def featureNormalize(X):
    """
    Normalizes the features in X. returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when working with
    learning algorithms.
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n).
    Returns
    -------
    X_norm : array_like
        The normalized dataset of shape (m x n).
    Instructions
    ------------
    First, for each feature dimension, compute the mean of the feature
    and subtract it from the dataset, storing the mean value in mu.
    Next, compute the  standard deviation of each feature and divide
    each feature by it's standard deviation, storing the standard deviation
    in sigma.
    Note that X is a matrix where each column is a feature and each row is
    an example. You needto perform the normalization separately for each feature.
    Hint
    ----
    You might find the 'np.mean' and 'np.std' functions useful.
    """
    # You need to set these values correctly
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])
    features = len(X[1, :])

    # =========================== YOUR CODE HERE =====================
    for f in range(features):
        mu[f] = (np.mean(X[:, f]))
        sigma[f] = (np.std(X[:, f]))
        X_norm[:, f] = (X_norm[:, f] - mu[f]) / sigma[f]
    # ================================================================
    return X_norm, mu, sigma


# call featureNormalize on the loaded data
X_norm, mu, sigma = featureNormalize(X)

print('Computed mean:', mu)
print('Computed standard deviation:', sigma)

# ====================== YOUR CODE HERE =======================
# pyplot.plot(X, y, 'ro', ms=10, mec='k')
# pyplot.ylabel('Profit in $10,000')
# pyplot.xlabel('Population of City in 10,000s')

# Add intercept term to X
X = np.concatenate([np.ones((m, 1)), X_norm], axis=1)


def hypothesis(theta, X):
    return np.dot(X, theta)


def computeCostMulti(X, y, theta):
    """
    Compute cost for linear regression with multiple variables.
    Computes the cost of using theta as the parameter for linear regression to fit the data points in X and y.
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).
    y : array_like
        A vector of shape (m, ) for the values at a given data point.
    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )
    Returns
    -------
    J : float
        The value of the cost function.
    Instructions
    ------------
    Compute the cost of a particular choice of theta. You should set J to the cost.
    """
    # Initialize some useful values
    m = y.shape[0]  # number of training examples

    # You need to return the following variable correctly
    J = 0

    # ======================= YOUR CODE HERE ===========================
    J = (1 / (2 * m)) * np.sum(np.square(hypothesis(theta, X) - y))
    # ==================================================================
    return J


J = computeCostMulti(X, y, theta=np.array([-1, 2, -3]))


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn theta.
    Updates theta by taking num_iters gradient steps with learning rate alpha.
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).
    y : array_like
        A vector of shape (m, ) for the values at a given data point.
    theta : array_like
        The linear regression parameters. A vector of shape (n+1, )
    alpha : float
        The learning rate for gradient descent.
    num_iters : int
        The number of iterations to run gradient descent.
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

    # make a copy of theta, which will be updated by gradient descent
    theta = theta.copy()

    J_history = []

    for i in range(num_iters):

        h = hypothesis(theta, X)

        for j in range(X.shape[1]):
            theta[j] = theta[j] - (alpha / m) * ((h - y).dot(X[:, j]))
        # =================================================================

        # save the cost J in every iteration
        J_history.append(computeCostMulti(X, y, theta))
        print(computeCostMulti(X, y, theta))

    return theta, J_history