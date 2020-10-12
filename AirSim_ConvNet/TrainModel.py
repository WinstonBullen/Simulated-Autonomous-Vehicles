from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input, concatenate
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, EarlyStopping
from keras.optimizers import Nadam
from keras.models import Model
import h5py
import os

from AirSim_ConvNet.Generator import DriveDataGenerator  # Provided task specific data generation
from AirSim_ConvNet.Cooking import checkAndCreateDir  # Provided task specific preprocessing

COOKED_DATA_DIR = 'data_cooked/'  # h5 data
MODEL_OUTPUT_DIR = 'model'

# Get the datasets in h5 (cooked) format and the corresponding number of data points in each
train_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'train.h5'), 'r')
num_train_images = train_dataset['image'].shape[0]
eval_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'eval.h5'), 'r')
num_eval_images = eval_dataset['image'].shape[0]
test_dataset = h5py.File(os.path.join(COOKED_DATA_DIR, 'test.h5'), 'r')
num_test_images = test_dataset['image'].shape[0]

batch_size = 32

data_generator = DriveDataGenerator(rescale=1./255., horizontal_flip=True, brighten_range=[0.0, 0.4])
train_generator = data_generator.flow(train_dataset['image'], train_dataset['previous_state'],
                                      train_dataset['label'], batch_size=batch_size,
                                      zero_drop_percentage=0.95, roi=[76, 135, 0, 255])
eval_generator = data_generator.flow(eval_dataset['image'], eval_dataset['previous_state'],
                                     eval_dataset['label'], batch_size=batch_size,
                                     zero_drop_percentage=0.95, roi=[76, 135, 0, 255])


[sample_batch_train_data, ] = next(train_generator)
image_input_shape = sample_batch_train_data[0].shape[1:]
state_input_shape = sample_batch_train_data[1].shape[1:]

# Stacks Conv Relu layers (TO SELF: see https://cs231n.github.io/convolutional-networks/ for explanation)
pic_input = Input(shape=image_input_shape)
img_stack = Conv2D(16, (3, 3), name="convolution0", padding='same', activation='relu')(pic_input)
img_stack = MaxPooling2D(pool_size=(2, 2))(img_stack)
img_stack = Conv2D(32, (3, 3), activation='relu', padding='same', name='convolution1')(img_stack)
img_stack = MaxPooling2D(pool_size=(2, 2))(img_stack)
img_stack = Conv2D(32, (3, 3), activation='relu', padding='same', name='convolution2')(img_stack)
img_stack = MaxPooling2D(pool_size=(2, 2))(img_stack)
img_stack = Flatten()(img_stack)
img_stack = Dropout(0.2)(img_stack)

state_input = Input(shape=state_input_shape)
merged = concatenate([img_stack, state_input])

# Fully connected layers
merged = Dense(64, activation='relu', name='dense0')(merged)
merged = Dropout(0.2)(merged)
merged = Dense(10, activation='relu', name='dense2')(merged)
merged = Dropout(0.2)(merged)
merged = Dense(1, name='output')(merged)

# Using Nadam to accelerate learning process
optimizer = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model = Model(inputs=[pic_input, state_input], outputs=merged)
model.compile(optimizer=optimizer, loss='mse')

model.summary()

# Set callbacks for post-epoch analysis
plateau_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001, verbose=1)
checkpoint_filepath = os.path.join(MODEL_OUTPUT_DIR, 'models', '{0}_model.{1}-{2}.h5'.format('model',
                                                                                             '{epoch:02d}',
                                                                                             '{val_loss:.7f}'))
checkAndCreateDir(checkpoint_filepath)
checkpoint_callback = ModelCheckpoint(checkpoint_filepath, save_best_only=True, verbose=1)
csv_callback = CSVLogger(os.path.join(MODEL_OUTPUT_DIR, 'training_log.csv'))
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
callbacks = [plateau_callback, csv_callback, checkpoint_callback, early_stopping_callback]

history = model.fit_generator(train_generator, steps_per_epoch=num_train_images//batch_size,
                              epochs=500, callbacks=callbacks, validation_data=eval_generator,
                              validation_steps=num_eval_images//batch_size, verbose=2)

[sample_batch_train_data, sample_batch_test_data] = next(train_generator)
predictions = model.predict([sample_batch_train_data[0], sample_batch_train_data[1]])
