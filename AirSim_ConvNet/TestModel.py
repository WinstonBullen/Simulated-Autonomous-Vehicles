from keras.models import load_model
import tensorflow as tf
import airsim
import glob

from AirSim_ConvNet.AirSimClientPython import *  # Provided task specific AirSim Python client

# Better GPU memory management
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

# Specify model path here, otherwise defaults to the model with the best loss
MODEL_PATH = 'specify_path'
if MODEL_PATH == 'specify_path':
    models = glob.glob('model/models/*.h5')
    best_model = max(models, key=os.path.getctime)
    MODEL_PATH = best_model

print('Using model {0} for testing.'.format(MODEL_PATH))
model = load_model(MODEL_PATH)

# Connects to the AirSim client in the Unreal Engine
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = CarControls()
print('Connected to AirSim in Unreal!')

# Set default sensor values
car_controls.steering = 0
car_controls.throttle = 0
car_controls.brake = 0

# Blank image frame
image_buf = np.zeros((1, 59, 255, 3))

# Blank sensor values
state_buf = np.zeros((1, 4))


# Gets the current frame from the AirSim client for evaluation
def get_image():
    image_response = client.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])[0]
    image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
    image_rgba = image1d.reshape(image_response.height, image_response.width, 3)
    return image_rgba[76:135, 0:255, 0:3].astype(float)


# Evaluates each frame from the AirSim client until program termination
while True:
    car_state = client.getCarState()

    if car_state.speed < 5:
        car_controls.throttle = 1.0
    else:
        car_controls.throttle = 0.0

    image_buf[0] = get_image()
    state_buf[0] = np.array([car_controls.steering, car_controls.throttle, car_controls.brake, car_state.speed])
    model_output = model.predict([image_buf, state_buf])
    car_controls.steering = round(0.5 * float(model_output[0][0]), 2)

    print('Sending steering = {0}, throttle = {1}'.format(car_controls.steering, car_controls.throttle))

    client.setCarControls(car_controls)
