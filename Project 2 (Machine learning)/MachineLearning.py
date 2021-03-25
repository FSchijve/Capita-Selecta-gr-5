# First, we import PyTorch and NumPy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
import keras
from tensorflow.keras import layers
import math
from save_slices import Dataset, XY_dataset

#%%
# Standard variables

# Preprocessed dataset
processed_data_path = r"C:\Users\Dell\Documents\Medical_Imaging\CSMI_TUE\preprocessed_data_prostate_extra"

# Number of classes
num_classes = 2

#%%
# Variables to change

image_side = 128
batch_size = 40

#%%

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

#%%
x_tot = Dataset(filename = os.path.join(processed_data_path,"images.txt"))
y_tot = Dataset(filename = os.path.join(processed_data_path,"masks.txt"))

image_side = x_tot.image_side

# If train-validation split has not yet been done:

x_tot.shuffle(1337)
y_tot.shuffle(1337)

validation_samples = round(len(x_tot)*0.3)

x_train = Dataset(image_side)
y_train = Dataset(image_side)
for i in range(0,len(x_tot)-validation_samples):
    x_train.addimage(x_tot.getpath(i))
    y_train.addimage(y_tot.getpath(i))

x_val = Dataset(image_side)
y_val = Dataset(image_side)
for i in range(len(x_tot)-validation_samples,len(x_tot)):
    x_val.addimage(x_tot.getpath(i))
    y_val.addimage(y_tot.getpath(i))

print('xtrain:', len(x_train))
print('ytrain:', len(y_train))
print('x_val:', len(x_val))
print('y_val:', len(y_val))

train_set = XY_dataset(x_train,y_train,batch_size)
val_set = XY_dataset(x_val,y_val)

# Free up RAM in case the model definition cells were run multiple times
#keras.backend.clear_session()

# Build model
model = get_model((image_side,image_side), num_classes)
#model.summary()

model.compile(loss=MeanSquaredError(), optimizer=Adam(), metrics=['accuracy'])

model.fit(train_set,
          batch_size=batch_size, # Only change batch size at top of file!
          epochs=20,
          verbose=True,
          validation_data=val_set,
          steps_per_epoch=math.floor(len(train_set)/batch_size), # Don't change steps_per_epoch!
          validation_steps=len(val_set)) # Don't change validation_steps!

val_set.set_end_evaluation(True)

score = model.evaluate(val_set, verbose=True)

print('Test loss:', score[0])
print('Test accuracy:', score[1])