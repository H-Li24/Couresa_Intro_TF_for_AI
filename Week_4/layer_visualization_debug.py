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
    epochs=1,
    verbose=2
)




import numpy as np
import os
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt

# Define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after
# the first.
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

print(img_path)

img = load_img(img_path, target_size=(300, 300))  # this is a PIL image
x = img_to_array(img)  # Numpy array with shape (300, 300, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 300, 300, 3)

# Scale by 1/255
x /= 255

# Run the image through the network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so you can have them as part of the plot
layer_names = [layer.name for layer in model.layers[1:]]

# Display the representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  if len(feature_map.shape) == 4:

    # Just do this for the conv / maxpool layers, not the fully-connected layers
    n_features = feature_map.shape[-1]  # number of features in feature map

    # The feature map has shape (1, size, size, n_features)
    size = feature_map.shape[1]
    
    # Tile the images in this matrix
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
      x = feature_map[0, :, :, i]
      x -= x.mean()
      # x /= x.std()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')
    
      # Tile each filter into this big horizontal grid
      display_grid[:, i * size : (i + 1) * size] = x
    
    # Display the grid
    scale = 20. / n_features
    plt.figure(figsize=(scale * n_features, scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')