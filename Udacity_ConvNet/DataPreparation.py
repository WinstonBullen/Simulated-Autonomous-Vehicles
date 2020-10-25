import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
import cv2
import pandas as pd
import random
import os
import ntpath
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
from imgaug import augmenters as iaa

data_dir = './data'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']

# Show head of csv file
data = pd.read_csv(os.path.join(data_dir, 'driving_log.csv'), names = columns)
pd.set_option('display.max_colwidth', -1)
print(data.head())

# Plot steering angle histogram
num_bins = 25
samples_per_bin = 400
hist, bins = np.histogram(data['steering'], num_bins)
center = (bins[:-1] + bins[1:]) * 0.5
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
plt.show()

def load_img_steering(data_dir, df):
    image_path = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
        image_path.append(os.path.join(data_dir, center.strip()))
        steering.append(float(indexed_data[3]))
        image_path.append(os.path.join(data_dir, left.strip()))
        steering.append(float(indexed_data[3]) + 0.15)
        image_path.append(os.path.join(data_dir, right.strip()))
        steering.append(float(indexed_data[3]) - 0.15)
    image_paths = np.asarray(image_path)
    steerings = np.asarray(steering)
    return image_paths, steerings

image_paths, steerings = load_img_steering(data_dir + '/IMG', data)

input_train, input_validation, target_train, target_validation = train_test_split(image_paths,
                                                                                  steerings,
                                                                                  test_size=0.2,
                                                                                  random_state=6)
print('Training Samples: ', len(input_train))
print('Validation Samples: ', len(input_validation))

def zoom(image):
    zoom = iaa.Affine(scale=(1, 1.3))
    image = zoom.augment_image(image)
    return image


def pan(image):
    pan = iaa.Affine(translate_percent={"x" : (-0.1, 0.1), "y" : (-0.1, 0.1)})
    image = pan.augment_image(image)
    return image

def random_brightness(image):
    brightness = iaa.Multiply((0.2, 1.2))
    image = brightness.augment_image(image)
    return image

def random_flip(image, steering_angle):
    image = cv2.flip(image, 1)
    steering_angle = -steering_angle
    return image, steering_angle

def random_augment(image, steering_angle):
    image = mpimg.imread(image)
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = random_brightness(image)
    if np.random.rand() < 0.5:
        image, steering_angle = random_flip(image, steering_angle)

    return image, steering_angle

def img_preprocess(image):
    image = image[60:135,:,:]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))
    image = image / 255
    return image

def batch_generator(image_paths, steering_angle, batch_size, isTraining):
    while True:
        batch_img = []
        batch_steering = []

        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
            if isTraining:
                img, steering = random_augment(image_paths[random_index], steering_angle[random_index])
            else:
                img = mpimg.imread(image_paths[random_index])
                steering = steering_angle[random_index]
            img = img_preprocess(img)
            batch_img.append(img)
            batch_steering.append(steering)
        yield (np.asarray(batch_img), np.asarray(batch_steering))

input_train_generated, target_train_generated = next (batch_generator(input_train, target_train, 1, 1))
input_validation_generated, target_validation_generated = next (batch_generator(input_validation, target_validation, 1, 0))
