from distutils.command.upload import upload
from re import X
import numpy as np
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload() # The image paths then get loaded into this list called uploaded.

for fn in uploaded.keys():
    path = '/content/' + fn
    img = image.load_img(path, target_size=(300,300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])

    classes = model.predict(images, batch_size = 10)
    print(classes[0])
    if classes[0] > 0.5:
        print(fn + " is a human")
    else:
        print(fn + "is a horse")


