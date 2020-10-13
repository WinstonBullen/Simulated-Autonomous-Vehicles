import json
import datetime

from DistributedRL.Share.scripts_downpour.app.airsim_client import *  # Provided AirSim script
from DistributedRL.Share.scripts_downpour.app.rl_model import RlModel  # Provided RL script

'''
 NOTE: Most of this code was provided but I made many quality changes. I have not
       run this script because I have not trained the model. Unfortunately, training
       the model would take days locally and the cloud model is pay walled, as well
       as the desired Unreal map.
'''

MODEL_FILENAME = ''

# Loads model as seen in CentralAlgorithm.py
model = RlModel(None, False)
with open(MODEL_FILENAME, 'r') as f:
    checkpoint_data = json.loads(f.read())
    model.from_packet(checkpoint_data['model'])

car_client = CarClient()
car_client.confirmConnection()
car_client.enableApiControl(True)
car_controls = CarControls()
print('Connection established!')


# Get simulator frames
def get_image(car_client):
    image_response = car_client.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])[0]
    image1d = np.frombuffer(image_response.image_data_uint8, dtype=np.uint8)
    image_rgba = image1d.reshape(image_response.height, image_response.width, 4)
    return image_rgba[76:135, 0:255, 0:3].astype(float)


def append_to_ring_buffer(item, buffer, buffer_size):
    if len(buffer) >= buffer_size:
        buffer = buffer[1:]
    buffer.append(item)
    return buffer


state_buffer = []
state_buffer_len = 4

# Runs the car for two seconds before running model
car_controls.steering = 0
car_controls.throttle = 1
car_controls.brake = 0
car_client.setCarControls(car_controls)
stop_run_time = datetime.datetime.now() + datetime.timedelta(seconds=2)
while datetime.datetime.now() < stop_run_time:
    time.sleep(0.01)
    state_buffer = append_to_ring_buffer(get_image(car_client), state_buffer, state_buffer_len)

while True:  # Run model
    state_buffer = append_to_ring_buffer(get_image(car_client), state_buffer, state_buffer_len)
    next_state, dummy = model.predict_state(state_buffer)
    next_control_signal = model.state_to_control_signals(next_state, car_client.getCarState())

    car_controls.steering = next_control_signal[0]
    car_controls.throttle = next_control_signal[1]
    car_controls.brake = next_control_signal[2]

    print('State = {0}, steering = {1}, throttle = {2}, brake = {3}'.format(next_state,
                                                                            car_controls.steering,
                                                                            car_controls.throttle,
                                                                            car_controls.brake))

    car_client.setCarControls(car_controls)

    time.sleep(0.1)
