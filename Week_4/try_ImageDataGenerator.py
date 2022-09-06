from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(300,300), # resize the images to the uniformed size
    batch_size=128,
    class_mode='binary'
)
# you should point it at the directory that contains sub-directory that contain your images.
# the name of the sub-directory will be the name of the labels for you images.

test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(300,300), # resize the images to the uniformed size
    batch_size=128,
    class_mode='binary'
)