from tabnanny import verbose
import tensorflow as tf
from tensorflow import keras

model = keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    keras.layers.Conv2D(16, (3,3), activation=tf.nn.relu, input_shape=(300,300,3)),
    keras.layers.MaxPooling2D(2,2),
    # This is the second convolution
    keras.layers.Conv2D(32,(3,3), activation=tf.nn.relu),
    keras.layers.MaxPooling2D(2,2),
    # This is the third convolution
    keras.layers.Conv2D(64,(3,3), activation=tf.nn.relu),
    keras.layers.MaxPooling2D(2,2),
    # This is the fourth convolution
    keras.layers.Conv2D(64,(3,3), activation=tf.nn.relu), 
    keras.layers.MaxPooling2D(2,2),
    # This is the fifth convolution
    keras.layers.Conv2D(64,(3,3), activation=tf.nn.relu),
    keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN 
    keras.layers.Flatten(),
    # 512 neuron hidden layer
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(1, activation='sigmoid') # more efficient than softmax in binary classification
])

model.summary()

print('\n')

from keras.optimizers import RMSprop

model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(learning_rate=0.001),
    metrics=['accuracy']
)

# ImageDataGenerator class allows you to instantiate generators of augmented image batches (and their labels) via .flow_from_directory(directory)
from keras.preprocessing.image import ImageDataGenerator

# Class interitance
class_train_datagen = ImageDataGenerator(rescale=1/255) # NO: scale=1/255

# Flow training images in batches of 128 using train_datagen generator
train_generator = class_train_datagen.flow_from_directory(
    './horse-or-human/', # This is the source directory for training images
    target_size=(300,300), # All images will be resized to 300x300
    batch_size=128,
    class_mode='binary' # Since we use binary_crossentropy loss, we need binary labels
)

validation_generator = class_train_datagen.flow_from_directory(
    './validation-horse-or-human/', # This is the source directory for training images
    target_size=(300,300), # All images will be resized to 300x300
    batch_size=32,
    class_mode='binary' # Since we use binary_crossentropy loss, we need binary labels
)

# Training

history = model.fit(
    train_generator,
    steps_per_epoch=8, # how many steps for loading the data with data generator?
    epochs=15,
    verbose=2,
    validation_data = validation_generator,
    validation_steps=8
)

import os

# Model Prediction
test_image_dir = os.path.join('./context')
test_image_name = os.listdir(test_image_dir)
test_image_path = [os.path.join(test_image_dir, fname) for fname in test_image_name[:]]

print(test_image_path)

import numpy as np
from keras.utils import load_img, img_to_array

for fname in test_image_name:
    path = os.path.join(test_image_dir, fname)
    img = load_img(path, target_size=(300,300))
    x = img_to_array(img)
    x /= 255
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(classes[0])

    if classes[0]>0.5:
        print(fname + " is a human")
    else:
        print(fname + " is a horse")