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

#  training data stored in arrays X, y
data = loadmat(os.path.join('Data', 'ex4data1.mat'))
X, y = data['X'], data['y'].ravel()

# set the zero digit to 0, rather than its mapped 10 in this dataset
# This is an artifact due to the fact that this dataset was used in
# MATLAB where there is no index 0
y[y == 10] = 0

# Number of training examples
m = y.size

# Setup the parameters you will use for this exercise
input_layer_size = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25  # 25 hidden units
num_labels = 10  # 10 labels, from 0 to 9

# Load the weights into variables Theta1 and Theta2
weights = loadmat(os.path.join('Data', 'ex4weights.mat'))

# Theta1 has size 25 x 401
# Theta2 has size 10 x 26
Theta1, Theta2 = weights['Theta1'], weights['Theta2']

# swap first and last columns of Theta2, due to legacy from MATLAB indexing,
# since the weight file ex3weights.mat was saved based on MATLAB indexing
Theta2 = np.roll(Theta2, 1, axis=0)

# Unroll parameters
nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])

Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                    (hidden_layer_size, (input_layer_size + 1)))

Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                    (num_labels, (hidden_layer_size + 1)))

# Setup some useful variables
m = y.size

# You need to return the following variables correctly
J = 0
Theta1_grad = np.zeros(Theta1.shape)
Theta2_grad = np.zeros(Theta2.shape)

# ====================== YOUR CODE HERE ======================
a1 = np.concatenate([np.ones((m, 1)), X], axis=1)

a2 = utils.sigmoid(a1.dot(Theta1.T))
a2 = np.concatenate([np.ones((a2.shape[0], 1)), a2], axis=1)

a3 = utils.sigmoid(a2.dot(Theta2.T))

y_matrix = y.reshape(-1)
y1 = y.reshape(-1)
y_matrix = np.eye(num_labels)[y_matrix]

temp1 = Theta1
temp2 = Theta2
cost = 0

# J = (-1 / m) * np.sum((np.log(a3) * y_matrix) + np.log(1 - a3) * (1 - y_matrix))

for c in range(num_labels):
    cost = cost + ((np.log(a3[c]) * y1(-1)) + np.log(1 - a3[c]) * (1 - y1))

# ================================================================
# Unroll gradients
# grad = np.concatenate([Theta1_grad.ravel(order=order), Theta2_grad.ravel(order=order)])
grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])
