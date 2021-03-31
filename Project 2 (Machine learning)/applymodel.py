from keras.models import model_from_json
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from skimage.filters import threshold_otsu
import keras
import tensorflow as tf
import os
import numpy as np
from save_slices import Dataset, XY_dataset
import PIL
import matplotlib.pyplot as plt

#%%
def dice_coefficient(y_true, y_pred):

    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return numerator / (denominator + tf.keras.backend.epsilon())

#%%
# Preprocessed dataset
processed_data_path = "C:/Users/Dell/Documents/Medical_Imaging/CSMI_TUE/preprocessed_data_original"
h5_file = "C:/Users/Dell/Documents/Medical_Imaging/CSMI_TUE/code/Capita-Selecta-gr-5/Project 2 (Machine learning)/Models/modelprostatetransfer5.h5"
json_file = "C:/Users/Dell/Documents/Medical_Imaging/CSMI_TUE/code/Capita-Selecta-gr-5/Project 2 (Machine learning)/Models/modelprostatetransfer5.json"

# Number of classes
num_classes = 2

#%%
# Variables to change

image_side = 128
batch_size = 60

#%%
# x_tot = Dataset(filename = os.path.join(processed_data_path,"images.txt"))
# y_tot = Dataset(filename = os.path.join(processed_data_path,"masks.txt"))

# image_side = x_tot.image_side

# # If train-validation split has not yet been done:

# x_tot.shuffle(1337)
# y_tot.shuffle(1337)

# validation_samples = round(len(x_tot)*0.01) #TODO change back

# x_train = Dataset(image_side)
# y_train = Dataset(image_side)
# for i in range(0,len(x_tot)-validation_samples):
#     x_train.addimage(x_tot.getpath(i))
#     y_train.addimage(y_tot.getpath(i))

# x_val = Dataset(image_side)
# y_val = Dataset(image_side)
# for i in range(len(x_tot)-validation_samples,len(x_tot)):
#     x_val.addimage(x_tot.getpath(i))
#     y_val.addimage(y_tot.getpath(i))
x_train = Dataset(filename = os.path.join(processed_data_path,"images_train.txt"))
y_train = Dataset(filename = os.path.join(processed_data_path,"masks_train.txt"))
x_val = Dataset(filename = os.path.join(processed_data_path,"images_val.txt"))
y_val = Dataset(filename = os.path.join(processed_data_path,"masks_val.txt"))
image_side = x_val.image_side

print('xtrain:', len(x_train))
print('ytrain:', len(y_train))
print('x_val:', len(x_val))
print('y_val:', len(y_val))

train_set = XY_dataset(x_train,y_train,"train",batch_size)
val_set = XY_dataset(x_val,y_val, "validation")

#%%
print("\nLoading model")
with open(json_file, 'r') as f:
    model = model_from_json(f.read())
model.load_weights(h5_file)

print("Compiling model")
model.compile(loss=BinaryCrossentropy(), optimizer=Adam(learning_rate=0.05), metrics=[dice_coefficient, 'accuracy'])

print("Creating dataset")
val_set = XY_dataset(x_train,y_train, "test")

print("Do predictions")
val_preds = model.predict(val_set)

threshold = 0.3
for i in range(len(val_preds)):
    print(f"Save image {i+1}/{len(val_preds)}")
    mask = val_preds[i,:,:,0]

    # if i < 86:
    #     plt.imshow(mask)
    #     plt.colorbar()
    #     plt.savefig(f"prediction_nonthresholded{i}.jpg")
    #     plt.show()

    otsu_threshold = threshold_otsu(mask)
    if otsu_threshold > threshold:
        mask = mask > otsu_threshold
    else:
        mask = mask > threshold
        
    np.save(f"prediction{i}.npy", mask)
    # if i < 86:
    #     plt.imshow(mask)
    #     plt.savefig(f"prediction{i}.jpg")
    #     plt.show()        
    
    img, mask = val_set[i]
    np.save(f"realmask{i}.npy",mask)
    if i < 86:
        plt.imshow(mask)
        #plt.savefig(f"realmask{i}.jpg")
        plt.show()