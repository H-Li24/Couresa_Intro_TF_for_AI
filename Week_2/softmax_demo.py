import numpy as np
import tensorflow as tf
from tensorflow import keras

inputs = np.array([[1, 3, 4, 2]], dtype=float)

inputs = tf.convert_to_tensor(inputs)
print(f'input to softmax function: {inputs.numpy()}')

outputs = keras.activations.softmax(inputs) # tensor
print(f'output to softmax function: {outputs.numpy()}')

sum = tf.reduce_sum(outputs) # I: tensor O: float
print(f'sum of outputs: {sum}')

prediction = np.argmax(outputs) # I: tensor O: float
print(f'class with highest prob: {prediction}')