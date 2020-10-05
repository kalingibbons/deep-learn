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

try:
    from spiral_training_data import generate_spiral_data
except:
    import Spiral_training_data

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


def one_hot(array):
    y = array.copy()
    oh = np.zeros((y.size, y.max() + 1))
    oh[np.arange(y.size), y] = 1
    return oh


true_labels = one_hot(true_labels).T
# sizes = [n_dims, n_hidden, n_classes]
# y = [y for y in sizes[1:]]
# x = [x for x in sizes[:-1]]

weight_2 = 0.01 * np.random.randn(n_hidden, n_dims)
bias_2 = np.zeros((n_hidden, 1))

weight_3 = 0.01 * np.random.randn(n_classes, n_hidden)
bias_3 = np.zeros((n_classes, 1))

# some hyperparameters
learn_rate = 1e-0  # learning rate / step size. Eta in formulas
reg_strength = 0*1e-3  # regularization strength. Lambda in formulas


# %%
def relu(vals):
    return np.maximum(0, vals)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def weighted_input(weights, inputs, bias):
    return weights @ inputs + bias


def update_parameter(parameter, parameter_partial):
    return parameter - learn_rate * parameter_partial


def add_regularization(weight, weight_partial):
    return weight_partial + (reg_strength / n_observation) * weight
    # return weight_partial + reg_strength * weight


def ss(arr):
    return np.sum(arr ** 2)


# %%
# gradient descent loop
X = np.transpose(data_mat)
n_observation = X.shape[1]
n_epochs = int(1e5)
cost_scaler = 0.5 * (reg_strength / n_observation)
# cost_scaler = 0.5 * (reg_strength / 1)

for epoch in range(n_epochs):

    # Compute weighted inputs and activations
    weight_inp_2 = weighted_input(weight_2, X, bias_2)
    activation_2 = sigmoid(weight_inp_2)

    weight_inp_3 = weighted_input(weight_3, activation_2, bias_3)
    activation_3 = sigmoid(weight_inp_3)

    # Compute the loss
    cost_y = 0.5 * ((true_labels - activation_3) ** 2)
    avg_cost_y = cost_y.sum() / n_observation
    cost_regularization = cost_scaler * np.sum((ss(weight_2), ss(weight_3)))
    cost = avg_cost_y + cost_regularization
    if epoch % 1000 == 0:   # % -  modulo
        print(f'Epoch {epoch:04d}: loss {cost:.5f}')

    # Compute the error term
    delta_3 = (activation_3 - true_labels) * sigmoid_prime(weight_inp_3)
    delta_3 /= n_observation

    # Use backprop to get the gradient of the parameters
    # First backprop into parameters W3 and b3
    weight_3_partial = delta_3 @ activation_2.T
    weight_3_partial = add_regularization(weight_3, weight_3_partial)
    bias_3_partial = delta_3.sum(axis=1, keepdims=True)

    # Next backprop into hidden layer
    delta_2 = (weight_3.T @ delta_3) * sigmoid_prime(weight_inp_2)

    # Finally into W,b
    weight_2_partial = delta_2 @ X.T
    weight_2_partial = add_regularization(weight_2, weight_2_partial)
    bias_2_partial = delta_2.sum(axis=1, keepdims=True)

    # Update weights and biases
    weight_2 = update_parameter(weight_2, weight_2_partial)
    bias_2 = update_parameter(bias_2, bias_2_partial)

    weight_3 = update_parameter(weight_3, weight_3_partial)
    bias_3 = update_parameter(bias_3, bias_3_partial)

# evaluate training set accuracy
activation_2 = sigmoid(weighted_input(weight_2, X, bias_2))
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
z2_grid = weighted_input(weight_2, X_grid, bias_2)
# a2_grid = np.maximum(0, np.dot(weight_2, X_grid) + bias_2)
a2_grid = sigmoid(z2_grid)
z3_grid = np.dot(weight_3, a2_grid) + bias_3
Z = np.argmax(z3_grid, axis=0)
Z = np.argmax(z3_grid, axis=0)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.winter, alpha=0.3)
plt.scatter(X[0, :], X[1, :], c=true_labels.argmax(axis=0), s=40, cmap=plt.cm.winter)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
fig.savefig('spiral_net.png')

end = time.time()
print(f'Time needed to run program: {end - start:.4f} seconds')

print(activation_3.shape)
print("")
print(activation_3)
print("")
print(true_labels.shape)
print("")
print(true_labels)
print("")
print("true_labels-activation_3")
print("")
print(true_labels-activation_3)
print("")
print("Incorrect Use of broadcasting!!!")
