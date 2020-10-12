import matplotlib.pyplot as plt
from PIL import Image
import os
import AirSim_ConvNet.Cooking as Cooking  # Provided task specific preprocessing

RAW_DATA_DIR = 'data_raw/'  # raw image and sensor data

COOKED_DATA_DIR = 'data_cooked/'  # h5 data

# Sub-folders of the raw data
DATA_FOLDERS = ['normal_1', 'normal_2', 'normal_3', 'normal_4', 'normal_5', 'normal_6',
                'swerve_1', 'swerve_2', 'swerve_3']

# Shows image ROI for training and normal/swerve drive comparison histogram.
explain = False

if explain:
    roi = Image.open('../visuals/Image_ROI.JPG')
    plt.title('Image ROI for training')
    plt.imshow(roi)
    plt.axis('off')
    plt.show()

    drive = Image.open('../visuals/Normal_vs_Swerve_Variation.JPG')
    plt.title('Normal vs Swerve Drive')
    plt.imshow(drive)
    plt.axis('off')
    plt.show()

# Preprocesses data with a 7:2:1 train:eval:test split using the provided preprocessing Cooking.py program
train_eval_test_split = [0.7, 0.2, 0.1]
all_raw_folders = [os.path.join(RAW_DATA_DIR, f) for f in DATA_FOLDERS]
Cooking.cook(all_raw_folders, COOKED_DATA_DIR, train_eval_test_split)
