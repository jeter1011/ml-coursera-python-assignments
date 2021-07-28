# used for manipulating directory paths

# Scientific and vector computation for python
import numpy as np

# Plotting library

# Optimization module in scipy
from scipy import optimize


# hypothesis for linear regression
def lin_hypothesis(theta, X):
    return np.dot(X, theta)


# Compute Cost for linear regression
def lin_computeCost(X, y, theta):
    # initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly
    J = 0

    J = (1 / (2 * m)) * np.sum((lin_hypothesis(theta, X) - y) ** 2)

    return J


# Gradient Descent for linear regression
def lin_gradientDescent(X, y, theta, alpha, num_iters):
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
        h = lin_hypothesis(theta, X)
        theta[0] = theta[0] - (alpha / m) * (np.sum(h - y))
        theta[1] = theta[1] - (alpha / m) * (np.dot((h - y), X[:, 1]))
        # ===================================================================
        # save the cost J in every iteration
        J_history.append(lin_computeCost(X, y, theta))
        # print(J_history)
    return theta, J_history


# Normal Equation for Linear Regression
def lin_normalEqn(X, y):
    """
    Computes the closed-form solution to linear regression using the normal equations.
    Parameters
    ----------
    X : array_like
        The dataset of shape (m x n+1).
    y : array_like
        The value at each data point. A vector of shape (m, ).
    Returns
    -------
    theta : array_like
        Estimated linear regression parameters. A vector of shape (n+1, ).
    Instructions
    ------------
    Complete the code to compute the closed form solution to linear
    regression and put the result in theta.
    Hint
    ----
    Look up the function `np.linalg.pinv` for computing matrix inverse.
    """
    theta = np.zeros(X.shape[1])

    # ===================== YOUR CODE HERE ============================

    theta = (np.linalg.pinv(np.transpose(X).dot(X))).dot(np.transpose(X)).dot(y)

    # =================================================================
    return theta


# featureNormalize for Linear Regression
def lin_featureNormalize(X):
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


# multi ComputeCost function for linear regression
def lin_computeCostMulti(X, y, theta):
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
    J = (1 / (2 * m)) * np.sum(np.square(lin_hypothesis(theta, X) - y))
    # ==================================================================
    return J


# Gradient Descent for multi paramaters Linear Regression
def lin_gradientDescentMulti(X, y, theta, alpha, num_iters):
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

        h = lin_hypothesis(theta, X)

        for j in range(X.shape[1]):
            theta[j] = theta[j] - (alpha / m) * ((h - y).dot(X[:, j]))
        # =================================================================

        # save the cost J in every iteration
        J_history.append(lin_computeCostMulti(X, y, theta))
        print(lin_computeCostMulti(X, y, theta))

    return theta, J_history


def sigmoid(z):
    # convert input to a numpy array
    z = np.array(z)

    # You need to return the following variables correctly
    g = np.zeros(z.shape)

    # ====================== YOUR CODE HERE ======================

    g = 1 / (1 + np.exp(-z))

    # =============================================================
    return g


# logistic regression hypothesis
def log_hypothesis(theta, X):
    return sigmoid(np.dot(X, theta.T))


# logistic regression Cost Function
def log_costFunction(theta, X, y):
    # Initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)

    # ====================== YOUR CODE HERE ======================

    J = (1 / m) * np.sum(
        (np.dot(-y, np.log(log_hypothesis(theta, X)))) - (np.dot((1 - y), (np.log(1 - log_hypothesis(theta, X))))))
    grad = (1 / m) * np.dot((log_hypothesis(theta, X) - y), X)
    # =============================================================
    return J, grad


# ------------------------------------------------------------------------------------------
# set options for optimize.minimize
options = {'maxiter': 400}

# see documention for scipy's optimize.minimize  for description about
# the different parameters
# The function returns an object `OptimizeResult`
# We use truncated Newton algorithm for optimization which is
# equivalent to MATLAB's fminunc
# See https://stackoverflow.com/questions/18801002/fminunc-alternate-in-numpy
res = optimize.minimize(log_costFunction,
                        # initial_theta,
                        # (X, y),
                        jac=True,
                        method='TNC',
                        options=options)

# the fun property of `OptimizeResult` object returns
# the value of costFunction at optimized theta
cost = res.fun

# the optimized theta is in the x property
theta = res.x


# Print theta to screen
# print('Cost at theta found by optimize.minimize: {:.3f}'.format(cost))
# print('Expected cost (approx): 0.203\n');

# print('theta:')
# print('\t[{:.3f}, {:.3f}, {:.3f}]'.format(*theta))
# print('Expected theta (approx):\n\t[-25.161, 0.206, 0.201]')
# ------------------------------------------------------------------------------------------


def predict(theta, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression.
    Computes the predictions for X using a threshold at 0.5
    (i.e., if sigmoid(theta.T*x) >= 0.5, predict 1)
    Parameters
    ----------
    theta : array_like
        Parameters for logistic regression. A vecotor of shape (n+1, ).
    X : array_like
        The data to use for computing predictions. The rows is the number
        of points to compute predictions, and columns is the number of
        features.
    Returns
    -------
    p : array_like
        Predictions and 0 or 1 for each row in X.
    Instructions
    ------------
    Complete the following code to make predictions using your learned
    logistic regression parameters.You should set p to a vector of 0's and 1's
    """
    m = X.shape[0]  # Number of training examples

    # You need to return the following variables correctly
    p = np.zeros(m)

    # ====================== YOUR CODE HERE ======================
    p = np.around(sigmoid(np.dot(X, theta.T)))
    # ============================================================
    return p
