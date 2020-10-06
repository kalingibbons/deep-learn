# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 16:12:25 2017

@author:
Stanford Course CS231n
Convolutional Neural Networks for Visual Recognition
"""
# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
def generate_spiral_data(n_classes=3,
                         n_dims=2,
                         n_points=100,
                         init_angle=(4 * np.pi / 3),
                         show_plots=False):
    """Generates the arrays for the spiral data example.

    Args:
        n_classes (int, optional): Number of classes. Defaults to 3.
        n_dims (int, optional): Dimensionality of the data. Defaults to 2.
        n_points (int, optional): Number of points per class. Defaults to 100.
        init_angle (float, optional): Initial angle in radians. Defaults to
            4/3 * pi.
        show_plots (bool, optional): Whether to show the initial data plot.
            Defaults to False.

    Returns:
        data_mat (float[array]): A matrix of data points with one example per
            row.
        labels (int[array]): An array of true labels for each datapoint. Entry
            values will fall between (0, n_classes - 1).
    """
    data_mat = np.zeros((n_points * n_classes, n_dims))  # one example per row
    labels = np.zeros(n_points * n_classes, dtype='uint8')
    radius = np.linspace(0.0, 1, n_points)  # np.linspace(start,stop,num)
    for j in range(n_classes):
        ix = range(n_points * j, n_points * (j + 1))
        th = (
            np.linspace(j * init_angle, (j + 1) * init_angle, n_points)
            + np.random.randn(n_points) * 0.2
        )
        data_mat[ix] = np.c_[radius * np.cos(th), radius * np.sin(th)]
        labels[ix] = j

    # lets visualize The Data:
    if show_plots:
        plt.scatter(data_mat[:, 0],
                    data_mat[:, 1],
                    c=labels,
                    s=40,
                    cmap=plt.cm.winter)
        # See https://matplotlib.org/examples/color/colormaps_reference.html
        plt.show()
        print("Xt.shape and y.shape"), print(data_mat.shape, labels.shape)
    return data_mat, labels
