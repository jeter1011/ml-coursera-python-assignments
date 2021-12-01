# used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot

# Optimization module in scipy
from scipy import optimize

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat

# library written for this exercise providing additional functions for assignment submission, and others
import utils

# Load from ex5data1.mat, where all variables will be store in a dictionary
data = loadmat(os.path.join('Data', 'ex5data1.mat'))

# Extract train, test, validation data from dictionary
# and also convert y's form 2-D matrix (MATLAB format) to a numpy vector
X, y = data['X'], data['y'][:, 0]
Xtest, ytest = data['Xtest'], data['ytest'][:, 0]
Xval, yval = data['Xval'], data['yval'][:, 0]

# m = Number of examples
m = y.size

# Plot training data
pyplot.plot(X, y, 'ro', ms=10, mec='k', mew=1)
pyplot.xlabel('Change in water level (x)')
pyplot.ylabel('Water flowing out of the dam (y)');


def linearRegCostFunction(X, y, theta, lambda_=0.0):
    """
    Compute cost and gradient for regularized linear regression
    with multiple variables. Computes the cost of using theta as
    the parameter for linear regression to fit the data points in X and y.

    Parameters
    ----------
    X : array_like
        The dataset. Matrix with shape (m x n + 1) where m is the
        total number of examples, and n is the number of features
        before adding the bias term.

    y : array_like
        The functions values at each datapoint. A vector of
        shape (m, ).

    theta : array_like
        The parameters for linear regression. A vector of shape (n+1,).

    lambda_ : float, optional
        The regularization parameter.

    Returns
    -------
    J : float
        The computed cost function.

    grad : array_like
        The value of the cost function gradient w.r.t theta.
        A vector of shape (n+1, ).

    Instructions
    ------------
    Compute the cost and gradient of regularized linear regression for
    a particular choice of theta.
    You should set J to the cost and grad to the gradient.
    """
    # Initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)

    h = np.dot(X, theta.T)

    temp = np.array(theta)
    temp[0] = 0

    # ====================== YOUR CODE HERE ======================

    J = (1 / (2 * m)) * np.sum(np.square(h - y)) + (lambda_ / (2 * m)) * np.sum(np.square(temp[1:]))

    grad[0] = (1 / m) * np.dot((h - y), X[:, 0])
    grad[1:] = (1 / m) * np.dot((h - y), X[:, 1:]) + (lambda_ / m) * theta[1:]
    # ============================================================
    return J, grad



def hypothesis(theta, X):
    return np.dot(X, theta)


theta = np.array([1, 1])
J, _ = linearRegCostFunction(np.concatenate([np.ones((m, 1)), X], axis=1), y, theta, 1)

print('Cost at theta = [1, 1]:\t   %f ' % J)
print('This value should be about 303.993192)\n' % J)

theta = np.array([1, 1])
J, grad = linearRegCostFunction(np.concatenate([np.ones((m, 1)), X], axis=1), y, theta, 1)

print('Gradient at theta = [1, 1]:  [{:.6f}, {:.6f}] '.format(*grad))
print(' (this value should be about [-15.303016, 598.250744])\n')

# -----------------------------------------------------------------------------------------------------


# add a columns of ones for the y-intercept
X_aug = np.concatenate([np.ones((m, 1)), X], axis=1)
theta = utils.trainLinearReg(linearRegCostFunction, X_aug, y, lambda_=0)

#  Plot fit over the data
pyplot.plot(X, y, 'ro', ms=10, mec='k', mew=1.5)
pyplot.xlabel('Change in water level (x)')
pyplot.ylabel('Water flowing out of the dam (y)')
pyplot.plot(X, np.dot(X_aug, theta), '--', lw=2)


# -----------------------------------------------------------------------------------------------------

def learningCurve(X, y, Xval, yval, lambda_=0):
    """
    Generates the train and cross validation set errors needed to plot a learning curve
    returns the train and cross validation set errors for a learning curve.

    In this function, you will compute the train and test errors for
    dataset sizes from 1 up to m. In practice, when working with larger
    datasets, you might want to do this in larger intervals.

    Parameters
    ----------
    X : array_like
        The training dataset. Matrix with shape (m x n + 1) where m is the
        total number of examples, and n is the number of features
        before adding the bias term.

    y : array_like
        The functions values at each training datapoint. A vector of
        shape (m, ).

    Xval : array_like
        The validation dataset. Matrix with shape (m_val x n + 1) where m is the
        total number of examples, and n is the number of features
        before adding the bias term.

    yval : array_like
        The functions values at each validation datapoint. A vector of
        shape (m_val, ).

    lambda_ : float, optional
        The regularization parameter.

    Returns
    -------
    error_train : array_like
        A vector of shape m. error_train[i] contains the training error for
        i examples.
    error_val : array_like
        A vecotr of shape m. error_val[i] contains the validation error for
        i training examples.

    Instructions
    ------------
    Fill in this function to return training errors in error_train and the
    cross validation errors in error_val. i.e., error_train[i] and
    error_val[i] should give you the errors obtained after training on i examples.

    Notes
    -----
    - You should evaluate the training error on the first i training
      examples (i.e., X[:i, :] and y[:i]).

      For the cross-validation error, you should instead evaluate on
      the _entire_ cross validation set (Xval and yval).

    - If you are using your cost function (linearRegCostFunction) to compute
      the training and cross validation error, you should call the function with
      the lambda argument set to 0. Do note that you will still need to use
      lambda when running the training to obtain the theta parameters.

    Hint
    ----
    You can loop over the examples with the following:

           for i in range(1, m+1):
               # Compute train/cross validation errors using training examples
               # X[:i, :] and y[:i], storing the result in
               # error_train[i-1] and error_val[i-1]
               ....
    """
    # Number of training examples
    m = y.size

    # You need to return these values correctly
    error_train = np.zeros(m)
    error_val = np.zeros(m)

    # ====================== YOUR CODE HERE ======================
    for i in range(1, m + 1):
        theta_Xt = utils.trainLinearReg(linearRegCostFunction, X[:i, :], y[:i], lambda_)
        error_train[i - 1], _ = linearRegCostFunction(X[:i, :], y[:i], theta_Xt, lambda_=0)
        error_val[i - 1], _ = linearRegCostFunction(Xval, yval, theta_Xt, lambda_=0)
        # Compute train/cross validation errors using training examples
        # X[:i, :] and y[:i], storing the result in
        # error_train[i-1] and error_val[i-1]

    # =============================================================
    return error_train, error_val


Xval_aug = np.concatenate([np.ones((yval.size, 1)), Xval], axis=1)
error_train, error_val = learningCurve(X_aug, y, Xval_aug, yval, lambda_=0)
#
# pyplot.plot(np.arange(1, m + 1), error_train, np.arange(1, m + 1), error_val, lw=2)
# pyplot.title('Learning curve for linear regression')
# pyplot.legend(['Train', 'Cross Validation'])
# pyplot.xlabel('Number of training examples')
# pyplot.ylabel('Error')
# pyplot.axis([0, 13, 0, 150])
#
# print('# Training Examples\tTrain Error\tCross Validation Error')
# for i in range(m):
#     print('  \t%d\t\t%f\t%f' % (i + 1, error_train[i], error_val[i]))
p = 8

X_poly = np.zeros((X.shape[0], p))

X_p = np.transpose(X_poly)

for i in range(1, p + 1):
    X_temp = np.reshape(np.array(X**i), (-1, 1))
    X_p[i - 1] = np.transpose(X_temp)

X_poly = np.transpose(X_p)