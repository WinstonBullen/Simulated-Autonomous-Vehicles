from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from keras.initializers import random_normal
from keras.optimizers import Adam
from keras.models import Model

import matplotlib.pyplot as plt

from DistributedRL.Share.scripts_downpour.app.airsim_client import *  # Provided AirSim script

'''
 NOTE: This script is merely for exploring the algorithm of the reward function and
       network architecture. The network resembles the AirSim_ConvNet architecture
       since the task is similar.
'''


def reward_function(car_state, road_points):

    if car_state.speed < 2:
        return 0  # Do not reward a stopped vehicle

    # Car position with (x, y) values
    position_key = bytes('position', encoding='utf8')
    x_val_key = bytes('x_val', encoding='utf8')
    y_val_key = bytes('y_val', encoding='utf8')

    car_point = np.array([car_state.kinematics_true[position_key][x_val_key],
                          car_state.kinematics_true[position_key][y_val_key], 0])

    # Initialize large distance for the first minimum function call
    distance = 100000

    # Get the distance to the center line and reward smaller distances
    for line in road_points:
        local_distance = 0
        length_squared = ((line[0][0] - line[1][0]) ** 2) + ((line[0][1] - line[1][1]) ** 2)
        if length_squared != 0:
            t = max(0, min(1, np.dot(car_point - line[0], line[1] - line[0]) / length_squared))
            proj = line[0] + (t * (line[1] - line[0]))
            local_distance = np.linalg.norm(proj - car_point)

        distance = min(distance, local_distance)

    # As the distance from the center line decreases, the reward function increases
    return math.exp(-(distance * 1.2))  # as x decreases, e^(-x) increases


def get_image(car_client):
    image_response = car_client.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])[0]
    image1d = np.frombuffer(image_response.image_data_uint8, dtype=np.uint8)
    image_rgba = image1d.reshape(image_response.height, image_response.width, 4)
    return image_rgba[76:135, 0:255, 0:3]


car_client = CarClient()
car_client.confirmConnection()
image = get_image(car_client)

# Show dash cam image
image = plt.imshow(image)


# The main model input.
def create_model():
    # Convolutional layers
    pic_input = Input(shape=(59, 255, 3))
    unfreeze_conv_layers = False  # False for transfer learning, True for custom models

    img_stack = Conv2D(16, (3, 3), name='convolution0', padding='same', activation='relu',
                       trainable=unfreeze_conv_layers)(pic_input)
    img_stack = MaxPooling2D(pool_size=(2, 2))(img_stack)
    img_stack = Conv2D(32, (3, 3), name='convolution1', padding='same', activation='relu',
                       trainable=unfreeze_conv_layers)(img_stack)
    img_stack = MaxPooling2D(pool_size=(2, 2))(img_stack)
    img_stack = Conv2D(32, (3, 3), name='convolution2', padding='same', activation='relu',
                       trainable=unfreeze_conv_layers)(img_stack)
    img_stack = MaxPooling2D(pool_size=(2, 2))(img_stack)
    img_stack = Flatten()(img_stack)
    img_stack = Dropout(0.2)(img_stack)

    # Fully connected layers
    img_stack = Dense(128, name='rl_dense', kernel_initializer=random_normal(stddev=0.01))(img_stack)
    img_stack = Dropout(0.2)(img_stack)
    output = Dense(5, name='rl_output', kernel_initializer=random_normal(stddev=0.01))(img_stack)

    action_model = Model(inputs=[pic_input], outputs=output)

    action_model.compile(optimizer=Adam(), loss='mean_squared_error')
    action_model.summary()
