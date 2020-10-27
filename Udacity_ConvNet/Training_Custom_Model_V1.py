import keras
import tensorflow as tf
import DataPreparation
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from keras.callbacks import EarlyStopping


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

data_dir = './hill_data'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'), names=columns)
image_paths, steerings = DataPreparation.load_img_steering(data_dir + '/IMG', data)

input_train, input_validation, target_train, target_validation = train_test_split(image_paths,
                                                                                  steerings,
                                                                                  test_size=0.2,
                                                                                  random_state=6)
num_train_examples = len(input_train)
num_validation_examples = len(input_validation)
batch_size = 128
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

history = model.fit_generator(DataPreparation.batch_generator(input_train, target_train, batch_size, 1),
                              steps_per_epoch=5 * num_train_examples//batch_size,
                              epochs=20,
                              callbacks=[early_stopping_callback],
                              validation_data=DataPreparation.batch_generator(input_validation, target_validation, batch_size, 0),
                              validation_steps=num_validation_examples//batch_size,
                              verbose=1,
                              shuffle=1)

model.save('./models/custom_model_v1_hill.h5')
