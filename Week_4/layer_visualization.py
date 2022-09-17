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

# Training

history = model.fit(
    train_generator,
    steps_per_epoch=8, # how many steps for loading the data with data generator?
    epochs=15,
    verbose=2
)

# To get a feel for what kind of features your CNN has learned, one fun thing to do is to visualize how an input get transformed as it goes through the model

# each row is the output of a layer, and each image in the row is a specific filter in that output feature map

import numpy as np
import random
import os
from keras.utils import load_img, img_to_array

# Define a new Model that will take an image as input
# and will output intermediate representations for all layers in the previous model after the first
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = tf.keras.models.Model(inputs = model.input, outputs = successive_outputs)

# Prepare a random input image from the training set
train_horse_dir = './horse-or-human/horses'
train_horse_names = os.listdir(train_horse_dir)
horse_img_files = [os.path.join(train_horse_dir, f) for f in train_horse_names]

train_human_dir = './horse-or-human/humans'
train_human_names = os.listdir(train_human_dir)
human_img_files = [os.path.join(train_human_dir, f) for f in train_human_names]

img_path = random.choice(horse_img_files + human_img_files)

img = load_img(img_path, target_size=(300, 300)) # this is a PIL image # TODO what is PIL image
x = img_to_array(img) # Numpy array with shape
x = x.reshape((1,) + x.shape) # Numpy array with shape (1, 300, 300, 3)

x /= 255

# Run the image through the network, thus obtaining all intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

layer_names = [layer.name for layer in model.layers[1:]] # TODO what does this line mean?

# print(layer_names)

# print(successive_feature_maps)

import matplotlib.pyplot as plt

for layer_name, feature_maps in zip(layer_names, successive_feature_maps):
    if len(feature_maps.shape)==4:

        # Just do this for the conv / maxpooling layers, not the fully-connected layers
        n_features = feature_maps.shape[-1] # number of features in feature maps

        # The feature map has shape (1, size, size, n_features)
        size = feature_maps.shape[1]

        # Tile the images in this matrix
        display_grid = np.zeros((size, size * n_features))
        for i in range(n_features):
            # print(x)
            x = feature_maps[0, :, :, i]
            x -= x.mean()
            # x /= x.std()
            x *= 64
            x += 128
            x = np.clip(x,0,255).astype('uint8')

            # Tile each filter into this big horizonal grid
            display_grid[:, i * size : (i+1) * size] = x
        
        # Display the grid
        scale = 20. / n_features
        plt.figure(figsize=(scale * n_features, scale))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

plt.show()