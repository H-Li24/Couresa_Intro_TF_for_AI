from gc import callbacks
import tensorflow as tf
from tensorflow import keras
import numpy as np

##### Exercise 8
threshold = 0.85

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') >= threshold): # class var in python can receive global var. Interesting.
            print(f"\nReached {threshold*100}% accuracy so cancelling training!") # \n , but not /n
            self.model.stop_training = True

fmnist = keras.datasets.fashion_mnist # data object
(train_images, train_labels), (test_images, test_labels) = fmnist.load_data() # 28x28 image; 09 numbers

'''
# Exercise 7: do not normalize the data
'''

train_images = train_images / 255 # devide an entire array train_images = train_images / 255 # devide an entire array
test_images = test_images / 255
test_images = test_images / 255

''' # firstly write this one
[keras.layers.Flatten(input_shape=(28,28)),
keras.layers.Dense(128, activation=tf.nn.relu),
keras.layers.Dense(10, activation=tf.nn.softmax)]
'''

model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),
# [keras.layers.Flatten() ##### Exercise 3
keras.layers.Dense(128, activation=tf.nn.relu),
# keras.layers.Dense(1024, activation=tf.nn.relu), ##### Exercise 2
# keras.layers.Dense(128, activation=tf.nn.relu), ##### Exercise 5
keras.layers.Dense(10, activation=tf.nn.softmax)])
# keras.layers.Dense(5, activation=tf.nn.softmax)]) ##### Exercise 4

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
# model.compile(optimizer="adam", loss="mean_squared_error", metrics=['accuracy']) # cannot use MSE in classification

# model.fit(train_images, train_labels, epochs=3)
# model.fit(train_images, train_labels, epochs=30) ##### Exercise 6

callbacks=myCallback()
model.fit(train_images, train_labels, epochs=3, callbacks=[callbacks]) ##### Exercise 8: early stop

model.evaluate(test_images, test_labels)

##### Exercise 1

index = 42

classifications = model.predict(test_images)
print(classifications[index])
print(f'Predicted class: {np.argmax(classifications[index])}')

print(f'Labeled class: {test_labels[index]}')



