# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 16:12:25 2017

@author: Stanford Course CS231n 
Convolutional Neural Networks for Visual Recognition
See https://cs231n.github.io/neural-networks-case-study/
"""
import time
start = time.time()

# initialize parameters randomly
h = 100 # size of hidden layer
W2 = 0.01 * np.random.randn(h,D)
b2 = np.zeros((h,1))
W3 = 0.01 * np.random.randn(K,h)
b3 = np.zeros((K,1))

# some hyperparameters
eta = 1e-0
lmbda = 1e-3 # regularization strength

# gradient descent loop
X = np.transpose(Xt)
n = X.shape[1]
for i in range(10000):
  
  # evaluate class scores, [N x K]
  a2 = np.maximum(0, np.dot(W2,X) + b2) # note: Broadcasting & ReLU activation
  z3 = np.dot(W3,a2) + b3
  
  # compute the class probabilities
  exp_z3 = np.exp(z3)
  exp_sum = np.sum(exp_z3, axis=0, keepdims=True)
  #print("exp_sum.shape"), print(exp_sum.shape)
  #exp_div = 1.0/exp_sum
#  a3 = exp_z3*exp_div
  a3 = exp_z3/np.sum(exp_z3, axis=0, keepdims=True) # [N x K]

  # compute the loss: average maximum likelihood cost (cross entropy) and regularization
  cost_y = -np.log(a3[y,range(n)])
  cost_y_avg  =  np.sum(cost_y)/n
  cost_W2W3 = 0.5*(lmbda/n)*np.sum(W2*W2) + 0.5*(lmbda/n)*np.sum(W3*W3)
  cost = cost_y_avg + cost_W2W3
  if i % 1000 == 0:   # % -  modulo
    print("iteration %d: loss %f" % (i, cost))
#  
  # compute the gradient on scores
  delta3 = a3
  delta3[y,range(n)] -= 1
  delta3 /= n
#  
  # backpropate the gradient to the parameters
  # first backprop into parameters W3 and b3
  dW3 = np.dot(delta3,a2.T)
  db3 = np.sum(delta3, axis=1, keepdims=True)
  # next backprop into hidden layer
  delta2 = np.dot(W3.T,delta3)
  # backprop the ReLU non-linearity
  delta2[a2 <= 0] = 0
  # finally into W,b
  dW2 = np.dot(delta2,X.T)
  db2 = np.sum(delta2, axis=1, keepdims=True)
  
  # add regularization gradient contribution
  dW3  += (lmbda/n)*W3
  dW2  += (lmbda/n)*W2
  
  # perform a parameter update
  W2 += -eta * dW2
  b2 += -eta * db2
  W3 += -eta * dW3
  b3 += -eta * db3
  
  # evaluate training set accuracy
a2 = np.maximum(0, np.dot(W2,X) + b2)
a3 = np.dot(W3,a2) + b3
predicted_class = np.argmax(a3, axis=0)
print('training accuracy: %.4f' % (np.mean(predicted_class == y)))
misses = np.argwhere(predicted_class != y)
#
# plot the resulting classifier
g = 0.02
x_min, x_max = X[0,:].min() - 1, X[0,:].max() + 1
y_min, y_max = X[1,:].min() - 1, X[1,:].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, g),
                     np.arange(y_min, y_max, g))     
print
print("xx.shape:"), print(xx.shape)
print
print("yy.shape:"), print(yy.shape)
print
#Zt = np.dot(np.maximum(0, np.dot(np.c_[xx.ravel(), yy.ravel()], W2t) + b2t), W3t) + b3t
Xt_grid = np.c_[xx.ravel(), yy.ravel()]
X_grid = np.transpose(Xt_grid)
a2_grid = np.maximum(0, np.dot(W2,X_grid) + b2)
z3_grid = np.dot(W3,a2_grid) + b3       
Z = np.argmax(z3_grid, axis=0)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[0,:], X[1,:], c=y, s=40, cmap=plt.cm.winter)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
fig.savefig('spiral_net.png')

origin = [X[:,0],X[:,100],X[:,200]]
z3_origin = np.asarray([z3[:,0],z3[:,100],z3[:,200]]).T
z3_argmax = np.argmax(z3_origin,axis=0)

end = time.time()
print('time needed to run program:', end - start)

