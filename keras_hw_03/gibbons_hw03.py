# %% [markdown]
# # Homework 03
#
# Kalin Gibbons


# %% [markdown]
# # Load MNIST Data
#

# %%
from matplotlib.pyplot import imshow
import numpy as np
import matplotlib.pyplot as plt

from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import regularizers, initializers


# %% [markdown]
# #  Load the data and plot the digits


# %%
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = train_images[1]
im_shape = digit.shape
n_train = train_images.shape[0]
n_test = test_images.shape[0]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
print(train_labels[0:10])


# %% [markdown]
# # Encode the image data
#
# Need to unravel the 2D image arrays into a 1D array.


# %%
train_images = train_images.reshape((n_train, np.prod(im_shape)))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((n_test, np.prod(im_shape)))
test_images = test_images.astype('float32') / 255
print('train_images.shape: {}'.format(train_images.shape))
print('train_images.ndim: {}'.format(train_images.ndim))


# %% [markdown]
# # Encode the Labels
#
# Need to convert the labels to one-hot encoding.


# %%
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print('train_labels.shape: {}'.format(train_labels.shape))
print('train_labels.ndim {}'.format(train_labels.ndim))
print(train_labels[0])


# %% [markdown]
# # The Network Architecture

# %%
network = models.Sequential()  # Specify layers in their sequential order
reg_strength = 0.001
hidden_layers = (
    layers.Dense(units=512,
                 activation='relu',
                 kernel_initializer=initializers.glorot_normal(),
                 kernel_regularizer=regularizers.l2(reg_strength),
                 input_shape=(np.prod(im_shape), )),
)
final_layer = layers.Dense(units=10, activation='softmax')

# Assemble the network architecture
for h_layer in hidden_layers:
    network.add(h_layer)
network.add(final_layer)


# %% [markdown]
# # Compile the Network


# %%
network.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# Options for  optimizer = 'adam', 'sgd'
# Other Options loss = 'mean_squared_error' or 'mse'


# %% [markdown]
# # Train the Network


# %%
network.fit(train_images, train_labels, epochs=5, batch_size=128)
#  "fit" refers to fitting the network weights to the data


# %% [markdown]
# # Check Accuracy on Test Data


# %%
network.evaluate(np.array(test_images),
                 np.array(test_labels),
                 batch_size=len(test_images))
