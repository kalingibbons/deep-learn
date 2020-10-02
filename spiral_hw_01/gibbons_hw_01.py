# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 16:12:25 2017

@author: Stanford Course CS231n
Convolutional Neural Networks for Visual Recognition
See https://cs231n.github.io/neural-networks-case-study/
"""


# %%
import time

import matplotlib.pyplot as plt
import numpy as np

from spiral_training_data import generate_spiral_data

start = time.time()


# %%
# initialize parameters randomly
try:
    (n_classes, n_dims, n_points, n_hidden) = (K, D, N, h)
    (data_mat, true_labels) = (Xt, y)

except NameError:
    n_classes = 3
    n_dims = 2
    n_points = 100
    n_hidden = 100  # size of hidden layer
    data_mat, true_labels = generate_spiral_data(n_classes=n_classes,
                                                 n_dims=n_dims,
                                                 n_points=n_points)

weight_2 = 0.01 * np.random.randn(n_hidden, n_dims)
bias_2 = np.zeros((n_hidden, 1))

weight_3 = 0.01 * np.random.randn(n_classes, n_hidden)
bias_3 = np.zeros((n_classes, 1))

# some hyperparameters
learn_rate = 1e-0  # learning rate / step size. Eta in formulas
reg_strength = 1e-3  # regularization strength. Lambda in formulas


# %%
def relu(vals):
    """Filter using Rectified Linear Unit

    Any value less than zero is replaced with zero. The rest remain unchanged.

    Args:
        vals (float): The values to be filtered.

    Returns:
        float: The filtered values, all >= 0.
    """
    return np.maximum(0, vals)


def weighted_input(weights, inputs, bias):
    return weights @ inputs + bias


def update_parameter(parameter, parameter_partial):
    return parameter - learn_rate * parameter_partial


def add_regularization(weight, weight_partial):
    return weight_partial + (reg_strength / n_feature) * weight


def sum_sqr(arr):
    return np.sum(arr ** 2)

# %%
# gradient descent loop
X = np.transpose(data_mat)
n_feature = X.shape[1]
n_epochs = int(1e4)
cost_scaler = 0.5 * (reg_strength / n_feature)
# TODO: make this into a function Loop just updates weights
for epoch in range(n_epochs):

    # evaluate class scores, [N x K]
    # note: Array broadcasting & ReLU activation
    weight_inp_2 = weighted_input(weight_2, X, bias_2)
    activation_2 = relu(weight_inp_2)
    weight_inp_3 = weighted_input(weight_3, activation_2, bias_3)

    # compute the class probabilities
    exp_weight_inp_3 = np.exp(weight_inp_3)
    exp_sum = np.sum(exp_weight_inp_3, axis=0, keepdims=True)
    activation_3 = exp_weight_inp_3 / exp_sum  # [N x K]

    # compute the loss: average maximum likelihood cost (cross entropy) and
    # regularization
    # TODO: Make this a function
    cost_y = -np.log(activation_3[true_labels, range(n_feature)])
    avg_cost_y = np.sum(cost_y) / n_feature
    cost_W2W3 = cost_scaler * np.sum((sum_sqr(weight_2), sum_sqr(weight_3)))
    cost = avg_cost_y + cost_W2W3
    if epoch % 1000 == 0:   # % -  modulo
        print(f'Epoch {epoch:04d}: loss {cost:.5f}')

    # compute the gradient on scores
    # TODO: Make this a function
    delta_3 = activation_3
    delta_3[true_labels, range(n_feature)] -= 1
    delta_3 /= n_feature

    # backpropogate the gradient to the parameters
    # first backpropogate into parameters W3 and b3
    # TODO: Make this a function until regularization
    weight_3_partial = delta_3 @ activation_2.T
    bias_3_partial = delta_3.sum(axis=1, keepdims=True)

    # next backprop into hidden layer
    delta_2 = weight_3.T @ delta_3

    # backprop the ReLU non-linearity
    delta_2[activation_2 <= 0] = 0

    # finally into W,b
    weight_2_partial = delta_2 @ X.T
    bias_2_partial = delta_2.sum(axis=1, keepdims=True)

    # add regularization gradient contribution
    weight_3_partial = add_regularization(weight_3, weight_3_partial)
    weight_2_partial = add_regularization(weight_2, weight_2_partial)

    # perform a parameter update
    weight_2 = update_parameter(weight_2, weight_2_partial)
    bias_2 = update_parameter(bias_2, bias_2_partial)

    weight_3 = update_parameter(weight_3, weight_3_partial)
    bias_3 = update_parameter(bias_3, bias_3_partial)

# evaluate training set accuracy
activation_2 = relu(weighted_input(weight_2, X, bias_2))
activation_3 = weighted_input(weight_3, activation_2, bias_3)
predicted_class = activation_3.argmax(axis=0)
print(f'training accuracy: {np.mean(predicted_class == true_labels):.4f}')
misses = np.argwhere(predicted_class != true_labels)
#
# plot the resulting classifier
g = 0.02
x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, g),
                     np.arange(y_min, y_max, g))

print(f'\nxx.shape: \n\t{xx.shape}\n')
print(f'yy.shape: \n\t{yy.shape}\n')

Xt_grid = np.c_[xx.ravel(), yy.ravel()]
X_grid = np.transpose(Xt_grid)
a2_grid = np.maximum(0, np.dot(weight_2, X_grid) + bias_2)
z3_grid = np.dot(weight_3, a2_grid) + bias_3
Z = np.argmax(z3_grid, axis=0)
Z = np.argmax(z3_grid, axis=0)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.winter, alpha=0.3)
plt.scatter(X[0, :], X[1, :], c=true_labels, s=40, cmap=plt.cm.winter)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
fig.savefig('spiral_net.png')

origin = [X[:, 0], X[:, 100], X[:, 200]]
z3_origin = np.asarray(
    [weight_inp_3[:, 0], weight_inp_3[:, 100], weight_inp_3[:, 200]]
).T
z3_argmax = np.argmax(z3_origin, axis=0)

end = time.time()
print(f'Time needed to run program: {end - start:.4f} seconds')
