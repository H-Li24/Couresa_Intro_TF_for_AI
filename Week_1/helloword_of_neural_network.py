import tensorflow as tf
from tensorflow import keras # bad: import tensorflow.keras as keras 
import numpy as np

print(tf.__version__)

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])]) # define
model.compile(optimizer='sgd', loss='mean_squared_error') # compile

xs = np.array([-1, 0 , 1, 2, 3, 4], dtype=float)
ys = np.array([-3, -1 , 1, 3, 5, 7], dtype=float)

model.fit(xs, ys, epochs=500)

print(model.predict([10]))

# Result is [[18.98722]]. Very close to 19.
