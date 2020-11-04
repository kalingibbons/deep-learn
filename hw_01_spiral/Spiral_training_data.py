# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 16:12:25 2017

@author: 
Stanford Course CS231n 
Convolutional Neural Networks for Visual Recognition
"""
import numpy as np
import matplotlib.pyplot as plt

N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
Xt = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  Xt[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
# t = np.linspace(j*(2*np.pi/3),(j+1)*(2*np.pi/3),N) + np.random.randn(N)*0.2 # theta
# Xt[ix] = np.c_[r*np.cos(t), r*np.sin(t)]
  y[ix] = j
# lets visualize the data:
plt.scatter(Xt[:, 0], Xt[:, 1], c=y, s=40, cmap=plt.cm.winter)
# See https://matplotlib.org/examples/color/colormaps_reference.html 
plt.show()
print("X.shape:", Xt.shape)
