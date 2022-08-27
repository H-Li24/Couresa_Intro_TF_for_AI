import tensorflow as tf
from tensorflow import keras
import numpy as np

fmnist = keras.datasets.fashion_mnist # data object
(train_images, train_labels), (test_images, test_labels) = fmnist.load_data() # 28x28 image; 09 numbers

'''
# normalize the data
'''

train_images = train_images / 255 # devide an entire array
test_images = test_images / 255

'''
# firstly write this one
[keras.layers.Flatten(input_shape=(28,28)),
keras.layers.Dense(128, activation=tf.nn.relu),
keras.layers.Dense(10, activation=tf.nn.softmax)]
'''



model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),
keras.layers.Dense(128, activation=tf.nn.relu),
keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
# model.compile(optimizer="adam", loss="mean_squared_error", metrics=['accuracy']) # cannot use MSE in classification

model.fit(train_images, train_labels, epochs=3)

model.evaluate(test_images, test_labels)

##### Exercise 1

index = 42

classifications = model.predict(test_images)
print(classifications[index])
print(f'Predicted class: {np.argmax(classifications[index])}')

print(f'Labeled class: {test_labels[index]}')