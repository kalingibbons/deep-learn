# %% [markdown]
#  # Create a Headline
# Create a new cell.
# Put the cursor in the new cell.
# Press Esc key.
# Press m key.
# Write your comment.
# Press shift+Enter keys.
#

# %% [markdown]
# # Load MNIST Data
#

# %%
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical


# %%
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# %% [markdown]
# # MNIST Data Format

# %%
print(train_images.shape)
print(len(train_images))
print(train_images.ndim)


# %%
# test_images[0]   # 28x28 digits in grey level 0-255


# %%
train_labels


# %%
print('test_images.shape:{}'.format(test_images.shape))
print('test_labels.ndim:{}'.format(test_images.ndim))
print('len(test_images):{}'.format(len(test_images)))


# %%
# print('test_image_sample  {}'.format(test_images[0]))
test_labels

# %% [markdown]
# #  Plot the Digits

# %%
digit = train_images[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
print(train_labels[0:10])

# %% [markdown]
# # Prepare the Images

# %%
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
print('train_images.shape: {}'.format(train_images.shape))
print('train_images.ndim: {}'.format(train_images.ndim))

# %% [markdown]
# # Prepare the Labels

# %%
train_labels = to_categorical(train_labels)  # Convert labels to 1-hot
test_labels = to_categorical(test_labels)    # Convert labels to 1-hot
print('train_labels.shape: {}'.format(train_labels.shape))
print('train_labels.ndim {}'.format(train_labels.ndim))
print(train_labels[0])

# %% [markdown]
# # The Network Architecture

# %%
network = models.Sequential()  # Specify layers in their sequential order
# inputs are vectors in R^28*28 = R^784
# Dense = Fully Connected.
# Hidden layer has 512 neurons with ReLU activations.
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
# Ouput layer uses softmax with 10 ouput neurons
# Assume there are 512 neurons going into the output layer
network.add(layers.Dense(10, activation='softmax'))  # sigmoid relu

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
