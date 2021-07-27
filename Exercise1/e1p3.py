import os

# Scientific and vector computation for python
import matplotlib
import numpy as np

# Plotting library
from jedi.api.refactoring import inline
from matplotlib import pyplot

# Load data
data = np.loadtxt(os.path.join('Data', 'ex1data2.txt'), delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.size
X = np.concatenate([np.ones((m, 1)), X], axis=1)


def normalEqn(X, y):
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


# Calculate the parameters from the normal equation
theta = normalEqn(X, y)

# Display normal equation's result
print('Theta computed from the normal equations: {:s}'.format(str(theta)))

# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================

price = np.dot([1, 1650, 3], theta)

# ============================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): ${:.0f}'.format(price))
