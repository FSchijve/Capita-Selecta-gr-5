import os
import random

import nibabel as nib
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adam
from nibabel.testing import data_path
from scipy import ndimage
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img

img_size = (256,256)
# num_classes = 3
# batch_size = 12
input_dir = "imagesTr"
target_dir = "labelsTr"

input_img_paths = sorted([os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.endswith(".gz") and not fname.startswith(".")])
target_img_paths = sorted([os.path.join(target_dir, fname) for fname in os.listdir(target_dir) if fname.endswith(".gz") and not fname.startswith(".")])

print(len(input_img_paths))
print(len(target_img_paths))

for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
    print(input_path, "|", target_path)

input_arrays_list = []
target_arrays_list = []

for x,y in zip(input_img_paths, target_img_paths):
    input_image = nib.load(x)
    target_image = nib.load(y)

    #take only the [:,:,:,0] and not [:,:,:,1]
    input_image_list = input_image.get_fdata()[:,:,:,0]
    target_image_list = target_image.get_fdata()

    input_arrays = np.asarray(input_image_list).astype(np.float32)
    target_arrays = np.asarray(target_image_list).astype(np.float32)
    #print(input_arrays.shape)
    #print(target_arrays.shape)

    input_arrays_reshaped = np.transpose(input_arrays, axes = [2,0,1])
    target_arrays_reshaped = np.transpose(target_arrays, axes = [2,0,1])

    for i in range(input_arrays_reshaped.shape[0]):
        input_arrays_list.append(np.resize(input_arrays_reshaped[i],(256,256,1)))
        target_arrays_list.append(np.resize(target_arrays_reshaped[i],(256,256,1)))
        

print(len(input_arrays_list))
print(len(target_arrays_list))

input_arrays_list = np.asarray(input_arrays_list) 
target_arrays_list = np.asarray(target_arrays_list)

validation_samples = 15
random.Random(1337).shuffle(input_arrays_list)
random.Random(1337).shuffle(target_arrays_list)
x_train = input_arrays_list[:-validation_samples]
y_train = target_arrays_list[:-validation_samples]


print('xtrain:', len(x_train))
print('ytrain:', len(y_train))

x_val = input_arrays_list[-validation_samples:]
y_val = target_arrays_list[-validation_samples:]
print('x_val:', len(x_val))
print('y_val:', len(y_val))

numberofimages = len(x_train)
x_train = x_train.reshape(numberofimages,256,256,1)
y_train = y_train.reshape(numberofimages,256,256,1)
y_val = y_val.reshape(15,256,256,1)
x_val = x_val.reshape(15,256,256,1)

print('input array type:', type(input_arrays_list))
print('target array type:', type(target_arrays_list))

print('input array type[0]:', type(input_arrays_list[0]))
print('target array type[0]:', type(target_arrays_list[0]))

# x_train = tuple(x_train)
# y_train = tuple(y_train)
# x_val = tuple(x_val)
# y_val = tuple(y_val)

# train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))

# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256,256,1)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2, activation='softmax'))

# model.summary()

num_classes=1

def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (1,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
model = get_model(img_size, num_classes)
model.summary()


model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=32,
          epochs=12,
          verbose=1,
          validation_data=(x_val, y_val))

score = model.evaluate(x_val, y_val, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
