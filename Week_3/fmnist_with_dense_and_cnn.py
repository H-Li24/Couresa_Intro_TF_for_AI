import tensorflow as tf
from tensorflow import keras

# Load the Fashion MNIST dataset
fmnist = keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = fmnist.load_data()

# Normalize the pixel values
training_images = training_images / 255.0
test_images = test_images / 255.0

'''
# Define the dense layer model

model = tf.keras.models.Sequential(
[keras.layers.Flatten(),
keras.layers.Dense(128, activation=tf.nn.relu),
keras.layers.Dense(10, activation=tf.nn.softmax)])

'''

# Define CNN model

model = tf.keras.models.Sequential([
    keras.layers.Conv2D(32, (3,3), activation=tf.nn.relu, input_shape=(28,28,1)), # The number of convolutions you want to generate.
    # keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu,input_shape=(28,28,1)), # In this senario, 64 > 32 + 32
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(32, (3,3), activation=tf.nn.relu),  # The value here is purely arbitrary but it's good to use powers of 2 starting from 32.
    keras.layers.MaxPooling2D(2,2),

    # You'll follow the convolution with a MaxPool2D layer which is designed to compress the image, 
    # while maintaining the content of the features that were highlighted by the convlution.

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.summary() # NO: model.summary

# Setup training parameters
 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # NO: metric NO: [accuracy]

threshold_acc = 0.85
threshold_loss = 0.4

class myCallback_acc(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') >= threshold_acc):
            print(f"\nReached {threshold_acc * 100}% loss so cancelling training!")
            self.model.stop_training = True

class myCallback_loss(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') < threshold_loss):
            print(f"\nReached {threshold_loss} loss so cancelling training!")
            self.model.stop_training = True


callbacks=myCallback_loss()
 
# Train the model
print(f'\nMODEL TRAINING:')
model.fit(training_images,training_labels, epochs=3, callbacks=[callbacks])
# model.fit(training_images,training_labels, epochs=20) # overfitting

# Evaluate on the test set
print(f'\nMODEL EVALUATION:')
model.evaluate(test_images, test_labels)



# Visualization the Conv and Pooling
import matplotlib.pyplot as plt

f, axarr = plt.subplots(3,4)

FIRST_IMAGE=0
SECOND_IMAGE=23
THIRD_IMAGE=28
CONVOLUTION_NUMBER = 1

layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)

for x in range(0,4):
  f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x] 
  # If model is only 64conv, then this error occur: 
  # IndexError: too many indices for array: array is 2-dimensional, but 4 were indexed 
  # Therefore, to use this code, you should have at least 2 conv layers
  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[0,x].grid(False)
  
  f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[1,x].grid(False)
  
  f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[2,x].grid(False)


plt.show() # Do not write plt.show() in the for loop

