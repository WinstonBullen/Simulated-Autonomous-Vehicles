import keras
import tensorflow as tf
import DataPreparation
from sklearn.model_selection import train_test_split
import pandas as pd
import os


def custom_model_v1():
    model = keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(16, kernel_size=3, padding='same', activation='relu', input_shape=(66, 200, 3)))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1))

    optimizer = tf.keras.optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='mse', optimizer=optimizer)
    return model


model = custom_model_v1()
print(model.summary())

data_dir = './data'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'), names=columns)
image_paths, steerings = DataPreparation.load_img_steering(data_dir + '/IMG', data)

input_train, input_validation, target_train, target_validation = train_test_split(image_paths,
                                                                                  steerings,
                                                                                  test_size=0.2,
                                                                                  random_state=6)

history = model.fit_generator(DataPreparation.batch_generator(input_train, target_train, 100, 1),
                              steps_per_epoch=400,
                              epochs=10,
                              validation_data=DataPreparation.batch_generator(input_validation, target_validation, 100, 0),
                              validation_steps=200,
                              verbose=1,
                              shuffle=1)

model.save('custom_model_v1.h5')
