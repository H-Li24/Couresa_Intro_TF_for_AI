import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fmnist = keras.datasets.fashion_mnist # data object
(train_images, train_labels), (test_images, test_labels) = fmnist.load_data() # 28x28 image; 09 numbers

index = 42 # for image to be visualized

np.set_printoptions(linewidth=320) # set number of characters per row when printing

print(f'LABEL: {train_labels[index]}') # print(f' ') can directly print tensor into terminal
print(f'\nIMAGE PIXEL ARRAY:\n {train_images[index]}')

plt.imshow(train_images[index], cmap='Greys') # plt.imshow just finishes drawing a picture instead of printing it.

plt.show() # If you want to print the picture, you just need to add plt.show.