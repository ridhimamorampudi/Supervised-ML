import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.activations import sigmoid
from lab_utils_common import dlc
from lab_neurons_utils import plt_prob_1d, sigmoidnp, plt_linear, plt_logistic
plt.style.use('./deeplearning.mplstyle')
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

#linear regression and house prices
X_train = np.array([[1.0], [2.0]], dtype=np.float32)           #(size in 1000 square feet)
Y_train = np.array([[300.0], [500.0]], dtype=np.float32)       #(price in 1000s of dollars)

fig, ax = plt.subplots(1,1)
ax.scatter(X_train, Y_train, marker='x', c='r', label="Data Points")
ax.legend( fontsize='xx-large')
ax.set_ylabel('Price (in 1000s of dollars)', fontsize='xx-large')
ax.set_xlabel('Size (1000 sqft)', fontsize='xx-large')
plt.show()

#initilize one layer
linear_layer = tf.keras.layers.Dense(units=1, activation = 'linear', )

#no weights
linear_layer.get_weights()

#needs to be 2d so reshape, will trigger weights
a1 = linear_layer(X_train[0].reshape(1,1))
print(a1)

#The result is a tensor (another name for an array) with a shape of (1,1) or one entry.   
#Now let's look at the weights and bias. These weights are randomly initialized to small numbers and the bias defaults to being initialized to zero.

w, b= linear_layer.get_weights()
print(f"w = {w}, b={b}")

set_w = np.array([[200]])
set_b = np.array([100])

# set_weights takes a list of numpy arrays
linear_layer.set_weights([set_w, set_b])
print(linear_layer.get_weights())

#should produce the same values
prediction_tf = linear_layer(X_train)
prediction_np = np.dot( X_train, set_w) + set_b

plt_linear(X_train, Y_train, prediction_tf, prediction_np)
