# %% [markdown]
#
# # Homework 04
#
# **Student**: Kalin Gibbons
#
# **Course Instructor**: John Chiasson
#
# **Author (TA)**: Ruthvik Vaila
#
# > Note:
# > * In this notebook we shall load a large `NumPy` array directly into RAM to
# >   train a model.
# >
# > * While the model is training keep an eye on the time taken and RAM usage
# >   of your machine.
# >
# > * Tested on `Python 3.7.5` with `Tensorflow 1.15.0` and `Keras 2.2.4`.
# >
# > * Tested on `Python 2.7.17` with `Tensorflow 1.15.3` and `Keras 2.2.4`.


# %% [markdown]
#
# ## First we must import and configure Keras and the Tensorflow backend
#


# %%
import gzip
import os
import pickle
import sys
import warnings
from pathlib import Path

import IPython
import IPython.display as display
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import (Conv2D,
                                     Dropout,
                                     Flatten,
                                     Dense,
                                     MaxPooling2D)
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.client import device_lib


# %% [markdown]
#
# ## Configure Keras
#
# Filtering the warning messages is important, because the Tensorflow team love
# to leave extensive deprecation messages all over their repository. We also
# need to configure tensorflow to use a CUDA enabled GPU. Specifically we need
# to:
#
# 1. Create a setting allowing the end user to toggle between GPU or CPU.
# 2. Force `tensorflow` to allocate a small amount of GPU memory, then increase
#    the allocation, as needed. Otherwise it will default to all available
#    memory, and compete with the computer display. See this [github
#    issue](https://github.com/tensorflow/tensorflow/issues/24828).
# 3. Leave in a toggle for using _eager-execution_, for debugging purposes.


# %%
print(sys.version)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # setting it to -1 hides the GPU.
warnings.filterwarnings(action='once')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# tf.compat.v1.enable_eager_execution()

# Tensorflow-GPU settings
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Print out some diagnostic information about the host machine
print(f'TF version:{tf.__version__}, Keras version:{keras.__version__}\n')
device_lib.list_local_devices()


# %% [markdown]
#
# ## Load the data
#
# > Note:
# > There are `60000` images in the training set and each image needs to be of
# > size `(28, 28, 1)` for `Keras`. The extra dimension in `(28, 28, 1)`
# > indicates number of channels. In this case we have `1` channel because it's
# > a gray scale image. In datasets like `CIFAR-10`, `CIFAR-100`, and
# > `ImageNet` images have `3` channels `(RGB)`.
#
# The EMNIST dataset is `latin1` encoded, and contains 28x28 images of
# 47 hand-written alphanumeric characters.


# %%
data_dir = Path.cwd().parent / 'data'
emnist_path = data_dir / 'emnist-balanced.pkl.gz'
with gzip.open(emnist_path, 'rb') as file:
    train_data, val_data, test_data = pickle.load(file, encoding='latin1')
    train_x, train_y = train_data
    val_x, val_y = val_data
    test_x, test_y = test_data


# %%
def summarize_shapes(locs=None):
    shapes = pd.DataFrame(
        dict(
            train_x=[train_x.shape],
            first_train=[train_x[0].shape],
            second_train=[train_x[1].shape],
            last_train=[train_x[-1].shape],
            train_y=[train_y.shape],
            val_x=[val_x.shape],
            val_y=[val_y.shape],
            test_x=[test_x.shape],
            test_y=[test_y.shape]
        ),
        index=['shapes']
    )
    df = shapes.loc[locs] if locs else shapes
    return df


summarize_shapes()


# %%
# combine train and validation data as Keras will split it internally
# (so long as we're not subclassing our own Layers?)
train_x = np.concatenate([train_x, val_x], axis=0)
train_y = np.concatenate([train_y, val_y], axis=0)
summarize_shapes()


# %%
train_x = train_x.reshape(train_x.shape[0], 28, 28, 1)
summarize_shapes()


# %%
test_x = test_x.reshape(test_x.shape[0], 28, 28, 1)
summarize_shapes()


# %% [markdown]
# ## One hot vectorize labels


# %%
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)
print('train_y[0]\n', train_y[0])
summarize_shapes()


# %% [markdown]
# ## Visualize the dataset.


# %%
subplot_kwargs = dict(
    xticks=range(0, 28, 7),
    yticks=range(0, 38, 7)
)
fig, axes = plt.subplots(2, 10, figsize=(20, 4), subplot_kw=subplot_kwargs)
fig.subplots_adjust(
    left=0.12,
    bottom=0.1,
    right=0.9,
    top=0.8,
    wspace=0.3,
    hspace=0.5
)
for idx, ax in enumerate(axes.flatten()):
    ax.imshow(train_x[idx, ..., 0])

plt.suptitle('First 20 Images of the Dataset')
plt.show()


# %% [markdown]
# ## Setup a small CNN model using `tf.keras.Sequential`
#
# * A simple convolutional neural network with the structure
# * `32c32p64c64p->200->10`
# * `Adam optimizer` and `Cross Entropy Loss` with a learning rate ($\alpha$)
#   set to `0.005`.


# %%
# Fraction of training data to be used for validation


# %%
def build_model():
    model = tf.keras.Sequential()

    # FIRST CONVOLUTION LAYER
    model = tf.keras.Sequential(
        [
            Conv2D(
                filters=64,  # 32
                kernel_size=(5, 5),
                strides=(1, 1),
                activation='relu',
                input_shape=(28, 28, 1),
            ),  # The 32 maps are 24x24   # 28-5+1 = 24
            # Dropout(0.1),
            MaxPooling2D(
                pool_size=(2, 2),
                strides=2
            ),  # The 32 pooled maps are 12x12
            Conv2D(
                filters=128,  # 64
                kernel_size=(3, 3),
                strides=(1, 1),
                activation='relu',
            ),  # The 64 maps are 10x10   (12-3+1 = 10)
            MaxPooling2D(
                pool_size=(2, 2),
                strides=2,
            ),  # The 64 pooled maps are 5x5
            Conv2D(
                filters=256,  # 128
                kernel_size=(3, 3),
                strides=(1, 1),
                activation='relu'
            ),  # The 128 maps are 3x3 (5 - 3 + 1)
            Dropout(0.1),
            Flatten(),  # 64x5x5 -> 1600 neuron layer (128x3x3 -> 1152)
            Dense(units=128, activation='relu'),
            Dropout(0.5),
            Dense(units=47, activation='softmax'),
        ]
    )

    return model


def compile_model(model, loss_fcn, optimizer='adam', lr=1e-3):
    if(optimizer == 'adam'):
        optim = tf.keras.optimizers.Adam(lr=lr)
    else:
        optim = tf.keras.optimizers.SGD(lr=lr)

    model.compile(optimizer=optim, loss=loss_fcn, metrics=['accuracy'])
    print(model.summary())
    return None


def fit_model(model, X, y, batch_size, epochs=3, callbacks=None, **fit_kwargs):
    history = model.fit(X,
                        y,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        **fit_kwargs)
    return history


# callback to keras to save best model weights
best_weights = "cnn_weights_best.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(best_weights,
                                                monitor='val_acc',
                                                verbose=1,
                                                save_best_only=True,
                                                mode='max')
stopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc',
                                            patience=5,
                                            restore_best_weights=False)

model = build_model()
loss_fcn = 'categorical_crossentropy'
optimizer = 'adam'
compile_model(model,
              optimizer=optimizer,
              lr=5e-4,
              loss_fcn='categorical_crossentropy')


# %% [markdown]
# ## Train the CNN


# %%

# history = fit_model(model,
#                     train_x,
#                     train_y,
#                     batch_size=32,
#                     epochs=3,
#                     callbacks=[checkpoint],
#                     validation_split=0.1)


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=5,
    # shear_range=10,
    zoom_range=0.2,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # zca_whitening=True,
    # zca_epsilon=1e-5,
    validation_split=0.1,
)
datagen.fit(train_x)

bs = 8
train_it = datagen.flow(train_x, train_y, subset='training', batch_size=bs)
val_it = datagen.flow(train_x, train_y, subset='validation', batch_size=bs)
history = model.fit_generator(generator=train_it,
                              steps_per_epoch=int(np.ceil(60e3 * 0.9 / bs)),
                              epochs=3,
                              validation_data=val_it,
                              validation_steps=int(np.ceil(60e3 * 0.1 / bs)),
                              callbacks=[checkpoint, stopping])

# %% [markdown]
# # Test the model at the end


# %%
model.evaluate(test_x, test_y, batch_size=bs)
# datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#     # zca_whitening=True,
#     # zca_epsilon=1e-5
# )
# datagen.fit(test_x)
# test_it = datagen.flow(test_x, test_y, batch_size=bs)
# model.evaluate_generator(test_it)


# %% [markdown]
# # Test at the best validation accuracy


# %%
model.load_weights(best_weights)
model.compile(optimizer=optimizer, loss=loss_fcn, metrics=['accuracy'])
model.evaluate(test_x, test_y, batch_size=bs)
# model.evaluate_generator(test_it)


# %% [markdown]
# # Restart the notebook to free up the `GPU` and `RAM`.


# %%
# IPython.Application.instance().kernel.do_shutdown(True)  # automatically
# restarts kernel
session.close()

