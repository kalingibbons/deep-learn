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


# %%
# initialize parameters randomly
try:
    (n_classes, n_dims, n_points, n_hidden) = (K, D, N, h)
    (data_mat, true_labels) = (Xt, y)

except NameError:
    from spiral_data import generate_spiral_data
    n_classes = 3
    n_dims = 2
    n_points = 100
    n_hidden = 100  # size of hidden layer
    data_mat, true_labels = generate_spiral_data(n_classes=n_classes,
                                                 n_dims=n_dims,
                                                 n_points=n_points)

# %%
def encode(array):
    y = array.copy()
    oh = np.zeros((y.size, y.max() + 1))
    oh[np.arange(y.size), y] = 1
    return oh.T


def decode(one_hot, axis=0):
    y = one_hot.copy()
    return y.argmax(axis=axis)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def weighted_input(weights, inputs, bias):
    return weights @ inputs + bias


def update_parameter(parameter, parameter_partial):
    return parameter - learn_rate * parameter_partial


def regularize_cost(strength, n_obs, weights):
    ss = [np.sum(w ** 2) for w in weights]
    return strength / 2 / n_obs * np.sum(ss)


def quadratic_cost(true, predictions, n_observation):
    return 0.5 * ((true - predictions) ** 2).sum() / n_observation


def calculate_cost(true, pred, n_obs, strength, weights):
    cost = (quadratic_cost(true, pred, n_obs)
            + regularize_cost(strength, n_obs, weights))
    return cost


def feed_forward(inp, weight, bias, activation=sigmoid):
    z = weighted_input(weight, inp, bias)
    act = activation(z)
    return z, act


def calc_err(true, pred, final_weight, n_obs, activation_prime=sigmoid_prime):
    return (pred - true) * activation_prime(final_weight) / n_obs


def backprop_err(later_err,
                 later_weight,
                 curr_weighted_input,
                 activation_prime=sigmoid_prime):
    return (later_weight.T @ later_err) * activation_prime(curr_weighted_input)


def backprop_bias(current_error, axis=1):
    return current_error.sum(axis=axis, keepdims=True)


def backprop_weight(err, inp, strength, n_obs, weight):
    return err @ inp.T + (strength / n_obs) * weight


def regularize_weight_prime(weight, weight_partial):
    return weight_partial + (reg_strength / n_obs) * weight


# %%
start = time.time()
true_labels = encode(true_labels)

weight_2 = 0.01 * np.random.randn(n_hidden, n_dims)
bias_2 = np.zeros((n_hidden, 1))

weight_3 = 0.01 * np.random.randn(n_classes, n_hidden)
bias_3 = np.zeros((n_classes, 1))

# some hyperparameters
learn_rate = 1e-0  # learning rate / step size. Eta in formulas
reg_strength = 0e-3  # regularization strength. Lambda in formulas


# %%
# gradient descent loop
X = np.transpose(data_mat)
n_obs = X.shape[1]
n_epochs = int(1e5)

weights = [weight_2, weight_3]
biases = [bias_2, bias_3]

for epoch in range(n_epochs):

    # --- Feed Forward ---
    # Compute weighted inputs and activations
    weight_inp_2, active_2 = feed_forward(X, weight_2, bias_2)
    weight_inp_3, active_3 = feed_forward(active_2, weight_3, bias_3)

    # --- Training Metrics ---
    # Compute and print the regularized quadratic cost
    cost = calculate_cost(
        true=true_labels,
        pred=active_3,
        n_obs=n_obs,
        strength=reg_strength,
        weights=(weight_2, weight_3)
    )
    if epoch % 1000 == 0:   # % -  modulo
        print(f'Epoch {epoch:04d}: loss {cost:.5f}')

    # ---- Backpropagation ----
    weights = [weight_2, weight_3]
    biases = [bias_2, bias_3]

    weighted_inputs = [weight_inp_2, weight_inp_3]
    activations = [X, active_2]
    delta = calc_err(true_labels, active_3, weight_inp_3, n_obs)
    for idx in reversed(range(len(weights))):
        a = activations[idx]
        w = weights[idx]
        b = biases[idx]
        z_prev = weighted_inputs[idx - 1]

        # Backpropagate partial derivatives
        weight_pprime = backprop_weight(delta, a, reg_strength, n_obs, w)
        bias_pprime = backprop_bias(delta, axis=1)

        # Update parameters
        weights[idx] = update_parameter(w, weight_pprime)
        biases[idx] = update_parameter(b, bias_pprime)
        if idx > 0:
            delta = backprop_err(delta, w, z_prev)

    weight_2, weight_3 = weights
    bias_2, bias_3 = biases


# --- Evaluation Metrics ---
# Evaluate training accuracy
predicted_labels = decode(active_3)
true_labels = decode(true_labels)
print(f'training accuracy: {np.mean(predicted_labels == true_labels):.4f}')
misses = np.argwhere(predicted_labels != true_labels)

# --- Evaluation Plots ---
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
a2_grid = sigmoid(z2_grid)
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

end = time.time()
print(f'Time needed to run program: {end - start:.4f} seconds')
