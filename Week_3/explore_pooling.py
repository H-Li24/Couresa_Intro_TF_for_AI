from scipy import misc

# load the ascent image
ascent_image = misc.ascent()

import numpy as np

# Copy image to a numpy array
image_transformed = np.copy(ascent_image) # Tag: why should do this

# Get the dimension of the image
size_x = image_transformed.shape[0]
size_y = image_transformed.shape[1]

new_x = int(size_x/2)
new_y = int(size_y/2)

# Create blank image with reduced dimentions
newImage = np.zeros((new_x, new_y)) # NO: np.zeros(new_x, new_y)

# Iterate over the image
for x in range(0, size_x, 2):
    for y in range(0, size_y, 2):

        # Store all the piexel calues in the (2,2) pool
        pixels = []
        pixels.append(image_transformed[x,y])
        pixels.append(image_transformed[x+1,y])
        pixels.append(image_transformed[x,y+1])
        pixels.append(image_transformed[x+1,y+1])

        newImage[int(x/2), int(y/2)] = max(pixels)


import matplotlib.pyplot as plt


plt.figure()
plt.subplot(1,2,1)
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(ascent_image)

plt.subplot(1,2,2)
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(newImage)

plt.show()

