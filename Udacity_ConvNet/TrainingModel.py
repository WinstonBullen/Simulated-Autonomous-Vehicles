import keras
import tensorflow as tf
from keras.optimizers import Adam
import DataPreparation
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import matplotlib.pyplot as plt


def custom_model():
    model = keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(24, kernel_size=5, strides=[2, 2], input_shape=(66, 200, 3), activation='elu'))
    model.add(tf.keras.layers.Conv2D(36, kernel_size=5, strides=[2, 2], activation='elu'))
    model.add(tf.keras.layers.Conv2D(48, kernel_size=5, strides=[2, 2], activation='elu'))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='elu'))
    model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='elu'))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='elu'))
    model.add(tf.keras.layers.Dense(50, activation='elu'))
    model.add(tf.keras.layers.Dense(10, activation='elu'))
    model.add(tf.keras.layers.Dense(1))

    optimizer = Adam(lr=1e-3)
    model.compile(loss='mse', optimizer=optimizer)
    return model


model = custom_model()
print(model.summary())

data_dir = './data'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'), names = columns)
image_paths, steerings = DataPreparation.load_img_steering(data_dir + '/IMG', data)

input_train, input_validation, target_train, target_validation = train_test_split(image_paths,
                                                                                  steerings,
                                                                                  test_size=0.2,
                                                                                  random_state=6)

history = model.fit_generator(DataPreparation.batch_generator(input_train, target_train, 100, 1),
                              steps_per_epoch=300,
                              epochs=10,
                              validation_data=DataPreparation.batch_generator(input_validation, target_validation, 100, 0),
                              validation_steps=200,
                              verbose=1,
                              shuffle=1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')

model.save('nvidia_model.h5')
