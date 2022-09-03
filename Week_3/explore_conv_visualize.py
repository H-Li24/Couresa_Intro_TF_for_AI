from scipy import misc

# load the ascent image
ascent_image = misc.ascent()

import matplotlib.pyplot as plt

# Visualize the image
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(ascent_image)

plt.show()

