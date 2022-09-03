from scipy import misc

# load the ascent image
ascent_image = misc.ascent()

import numpy as np

# Copy image to a numpy array
image_transformed = np.copy(ascent_image) # Tag: why should do this

# Get the dimension of the image
size_x = image_transformed.shape[0]
size_y = image_transformed.shape[1]

filter = [[-1,0,1],[-2,0,2],[-1,0,1]] # Sobel filter vertical

Weight = 1 # normalize the filter

for x in range(1,size_x-1):
    for y in range(1,size_y-1):
        convolution = 0.0
        convolution = convolution + (ascent_image[x-1, y-1] * filter[0][0])
        convolution = convolution + (ascent_image[x-1, y] * filter[0][1])
        convolution = convolution + (ascent_image[x-1, y+1] * filter[0][2])
        convolution = convolution + (ascent_image[x, y-1] * filter[1][0])
        convolution = convolution + (ascent_image[x, y] * filter[1][1])
        convolution = convolution + (ascent_image[x, y+1] * filter[1][2])
        convolution = convolution + (ascent_image[x+1, y-1] * filter[2][0])
        convolution = convolution + (ascent_image[x+1, y] * filter[2][1])
        convolution = convolution + (ascent_image[x+1, y+1] * filter[2][2])

        # Multiply by weight
        convolution = convolution * Weight

        # Check the boundaries of the pixel value
        if(convolution<0):
            convolution=0
        if(convolution>255):
            convolution=255
        
        image_transformed[x,y] = convolution

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
plt.imshow(image_transformed)

plt.show()

