"""

 The following reward function assumes the following DeepRacer action space:
    Speed values [1, 2, 3] m/s
    Steering angles [-30, -20, -10, 0, 10, 20, 30] degrees

"""


def reward_function(params):
    track_width = params['track_width']
    distance_from_center = params['distance_from_center']
    steering = params['steering_angle']
    speed = params['speed']

    # Discrete distances from the center line
    marker_1 = 0.1 * track_width
    marker_2 = 0.25 * track_width
    marker_3 = 0.5 * track_width

    # Encourage staying close to the center line by increasing reward the closer
    # the car is to the center line.
    if distance_from_center <= marker_1:
        reward = 1.0
    elif distance_from_center <= marker_2:
        reward = 0.5
    elif distance_from_center <= marker_3:
        reward = 0.1
    else:
        reward = 1e-3

    # Discourage sharp turns by decreasing reward if sharp turns are taken
    if abs(steering) > 25:
        reward *= 0.5
    elif abs(steering) > 15:
        reward *= 0.75

    # Encourage faster speeds by increasing reward for faster speeds
    if speed > 2:
        reward *= 1.2
    elif speed > 1:
        reward *= 1.1

    return float(reward)
