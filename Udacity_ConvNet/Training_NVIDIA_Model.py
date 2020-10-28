import keras
import tensorflow as tf
from keras.optimizers import Adam
import DataPreparation
from sklearn.model_selection import train_test_split
import pandas as pd
import os

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
batch_size = 100
num_epochs = 10


def nvidia_model():
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


model = nvidia_model()
print(model.summary())

history = model.fit_generator(DataPreparation.batch_generator(input_train, target_train, batch_size, 1),
                              steps_per_epoch=num_train_examples * 3 // batch_size,
                              epochs=num_epochs,
                              validation_data=DataPreparation.batch_generator(input_validation, target_validation, batch_size, 0),
                              validation_steps=num_validation_examples * 3 // batch_size,
                              verbose=1,
                              shuffle=1)

model.save('./models/nvidia_model.h5')
