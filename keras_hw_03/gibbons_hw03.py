# %% [markdown]
# # Homework 03
#
# Kalin Gibbons


# %% [markdown]
# # Load MNIST Data
#

# %%
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import regularizers, initializers
from keras.wrappers.scikit_learn import KerasClassifier


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

def build_model(reg_strength=0, drop_rate=0, image_shape=im_shape):
    network = models.Sequential()  # Specify layers in their sequential order
    hidden_layers = (
        layers.Dense(units=512,
                     activation='relu',
                     kernel_initializer=initializers.glorot_normal(),
                     kernel_regularizer=regularizers.l2(reg_strength),
                     input_shape=(np.prod(image_shape), )),
        layers.Dropout(drop_rate)
    )
    final_layer = layers.Dense(units=10, activation='softmax')

    # Assemble the network architecture
    for h_layer in hidden_layers:
        network.add(h_layer)
    network.add(final_layer)

    network.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return network


# %% [markdown]
# # Perform Grid Search
seed = 42
np.random.seed(seed)
network = KerasClassifier(build_fn=build_model, verbose=0)

batch_size = [128]
epochs = [5]

# First round
drop_rate = [0, 0.2, 0.4, 0.6]
reg_strength = [0, 0.1, 1, 10.]
# Best: 0.9732499999999998 using {'batch_size': 128, 'drop_rate': 0.2, 'epochs': 5, 'reg_strength': 0}
# 0.971517 (0.001756) with {'batch_size': 128, 'drop_rate': 0, 'epochs': 5, 'reg_strength': 0}
# 0.872033 (0.015575) with {'batch_size': 128, 'drop_rate': 0, 'epochs': 5, 'reg_strength': 0.1}
# 0.829367 (0.016711) with {'batch_size': 128, 'drop_rate': 0, 'epochs': 5, 'reg_strength': 1}
# 0.688550 (0.047755) with {'batch_size': 128, 'drop_rate': 0, 'epochs': 5, 'reg_strength': 10.0}
# 0.973250 (0.001404) with {'batch_size': 128, 'drop_rate': 0.2, 'epochs': 5, 'reg_strength': 0}
# 0.885550 (0.009727) with {'batch_size': 128, 'drop_rate': 0.2, 'epochs': 5, 'reg_strength': 0.1}
# 0.839117 (0.003696) with {'batch_size': 128, 'drop_rate': 0.2, 'epochs': 5, 'reg_strength': 1}
# 0.658967 (0.098422) with {'batch_size': 128, 'drop_rate': 0.2, 'epochs': 5, 'reg_strength': 10.0}
# 0.971717 (0.001382) with {'batch_size': 128, 'drop_rate': 0.4, 'epochs': 5, 'reg_strength': 0}
# 0.889317 (0.011750) with {'batch_size': 128, 'drop_rate': 0.4, 'epochs': 5, 'reg_strength': 0.1}
# 0.825200 (0.011704) with {'batch_size': 128, 'drop_rate': 0.4, 'epochs': 5, 'reg_strength': 1}
# 0.734000 (0.022266) with {'batch_size': 128, 'drop_rate': 0.4, 'epochs': 5, 'reg_strength': 10.0}
# 0.969117 (0.002122) with {'batch_size': 128, 'drop_rate': 0.6, 'epochs': 5, 'reg_strength': 0}
# 0.876117 (0.014493) with {'batch_size': 128, 'drop_rate': 0.6, 'epochs': 5, 'reg_strength': 0.1}
# 0.798567 (0.002724) with {'batch_size': 128, 'drop_rate': 0.6, 'epochs': 5, 'reg_strength': 1}
# 0.670100 (0.040648) with {'batch_size': 128, 'drop_rate': 0.6, 'epochs': 5, 'reg_strength': 10.0}

# Second round
drop_rate = np.linspace(0, 0.25, 4).tolist()
reg_strength = [0] + np.logspace(-4, -2, 3).tolist()
# Best: 0.9731500000000001 using {'batch_size': 128, 'drop_rate': 0.08333333333333333, 'epochs': 5, 'reg_strength': 0}
# 0.971817 (0.001618) with {'batch_size': 128, 'drop_rate': 0.0, 'epochs': 5, 'reg_strength': 0}
# 0.970717 (0.001111) with {'batch_size': 128, 'drop_rate': 0.0, 'epochs': 5, 'reg_strength': 0.0001}
# 0.964983 (0.000946) with {'batch_size': 128, 'drop_rate': 0.0, 'epochs': 5, 'reg_strength': 0.001}
# 0.939850 (0.004624) with {'batch_size': 128, 'drop_rate': 0.0, 'epochs': 5, 'reg_strength': 0.01}
# 0.973150 (0.001064) with {'batch_size': 128, 'drop_rate': 0.08333333333333333, 'epochs': 5, 'reg_strength': 0}
# 0.972200 (0.001080) with {'batch_size': 128, 'drop_rate': 0.08333333333333333, 'epochs': 5, 'reg_strength': 0.0001}
# 0.963817 (0.001376) with {'batch_size': 128, 'drop_rate': 0.08333333333333333, 'epochs': 5, 'reg_strength': 0.001}
# 0.941267 (0.002299) with {'batch_size': 128, 'drop_rate': 0.08333333333333333, 'epochs': 5, 'reg_strength': 0.01}
# 0.972033 (0.000047) with {'batch_size': 128, 'drop_rate': 0.16666666666666666, 'epochs': 5, 'reg_strength': 0}
# 0.971550 (0.002466) with {'batch_size': 128, 'drop_rate': 0.16666666666666666, 'epochs': 5, 'reg_strength': 0.0001}
# 0.965100 (0.001073) with {'batch_size': 128, 'drop_rate': 0.16666666666666666, 'epochs': 5, 'reg_strength': 0.001}
# 0.941467 (0.000471) with {'batch_size': 128, 'drop_rate': 0.16666666666666666, 'epochs': 5, 'reg_strength': 0.01}
# 0.972433 (0.001073) with {'batch_size': 128, 'drop_rate': 0.25, 'epochs': 5, 'reg_strength': 0}
# 0.971800 (0.001337) with {'batch_size': 128, 'drop_rate': 0.25, 'epochs': 5, 'reg_strength': 0.0001}
# 0.961500 (0.002376) with {'batch_size': 128, 'drop_rate': 0.25, 'epochs': 5, 'reg_strength': 0.001}
# 0.937383 (0.006111) with {'batch_size': 128, 'drop_rate': 0.25, 'epochs': 5,
# 'reg_strength': 0.01}

param_grid = dict(batch_size=batch_size,
                  epochs=epochs,
                  drop_rate=drop_rate,
                  reg_strength=reg_strength)

grid = GridSearchCV(estimator=network,
                    param_grid=param_grid,
                    n_jobs=-1,
                    cv=3,
                    refit=False)

grid_result = grid.fit(train_images, train_labels)


# %%
print(f'Best: {grid_result.best_score_} using {grid_result.best_params_}')
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f'{mean:.6f} ({stdev:.6f}) with {param}')


# %% [markdown]
# # Check Accuracy on Test Data


# %%
network = build_model(reg_strength=0, drop_rate=0.2)
network.fit(train_images, train_labels, epochs=5, batch_size=128)
network.evaluate(np.array(test_images),
                 np.array(test_labels),
                 batch_size=len(test_images))


# %%
network.get_config()