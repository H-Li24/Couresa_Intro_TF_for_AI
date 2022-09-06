from tensorflow import keras

model=keras.models.Sequential([keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
keras.layers.MaxPooling2D(2,2),
keras.layers.Conv2D(64, (3,3), activation='relu'),
keras.layers.MaxPooling2D(2,2),
keras.layers.Flatten(),
keras.layers.Dense(128, activation='relu'),
keras.layers.Dense(10, activation='softmax')])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) # NO: metric NO: [accuracy]

model.fit(training_images,training_labels, epochs=3, callbacks=[callbacks])