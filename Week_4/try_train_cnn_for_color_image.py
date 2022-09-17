from tensorflow import keras
import tensorflow as tf

model = keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    keras.layers.Conv2D(16, (3,3), activation=tf.nn.relu, input_shape=(150,150,3)),
    keras.layers.MaxPooling2D(2,2),
    # This is the second convolution
    keras.layers.Conv2D(32,(3,3), activation=tf.nn.relu),
    keras.layers.MaxPooling2D(2,2),
    # This is the third convolution
    keras.layers.Conv2D(64,(3,3), activation=tf.nn.relu),
    keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN 
    keras.layers.Flatten(),
    # 512 neuron hidden layer
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(1, activation='sigmoid') # more efficient than softmax in binary classification
])

'''
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
'''

model.summary()

from keras.optimizers import RMSprop

model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(lr=0.001),
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    steps_per_epoch=8,# Total = 1024, batch_size = 128, steps_per_epoch = 1024 / 128 = 8 (number of batchs)
    epochs=15,
    validation_data=validation_generator,
    validation_steps=8, # Total = 256, batch_size = 32, steps_per_epoch = 256 / 32 = 8
    verbose=2 # specifies how much to display while training is going on 
)