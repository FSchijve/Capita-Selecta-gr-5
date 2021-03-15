import matplotlib.pyplot as plt
import os
import imageio
import numpy as np
import nibabel as nib
from sklearn import preprocessing

path_original_data = r"C:\Users\s160518\Documents\8DM20 Project\TrainingData" 
path_extra_data = r"C:\Users\s160518\Documents\CSMIA Segmentation\Task05_Prostate"

list_original_names = [102, 107, 108, 109, 115, 116, 117, 119, 120, 125, 127, 128, 129, 133, 135]
   
for i in range(len(list_original_names)):
    number = list_original_names[i]
    Images = imageio.imread(os.path.join(path_original_data,f"p{number}\mr_bffe.mhd"))
    Masks = imageio.imread(os.path.join(path_original_data,f"p{number}\mr_bffe.mhd"))
    Images = Images/np.amax(Images)
    Masks = Masks/np.amax(Masks)
    start_x =  int(Images.shape[1]/2-128)
    stop_x = int(Images.shape[1]/2+128)
    start_y = int(Images.shape[2]/2-128)
    stop_y = int(Images.shape[2]/2+128)
    Images = Images[:,start_x:stop_x,start_y:stop_y]
    Masks = Masks[:,start_x:stop_x,start_y:stop_y]
 
    locals()["image_p%s" % number] = Images
    locals()["mask_p%s" % number] = Masks
    
list_extra_names = [0, 1, 2, 4, 6, 7, 10, 13, 14, 16, 17, 18, 20, 21, 24, 25, 28, 29, 31, 32, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47]

for j in range(len(list_extra_names)):
    number = list_extra_names[j]
    Images = np.rot90(nib.load(os.path.join(path_extra_data,f"imagesTr\prostate_{number}.nii.gz")).get_data()[:,:,:,0])
    Masks = np.rot90(nib.load(os.path.join(path_extra_data,f"labelsTr\prostate_{number}.nii.gz")).get_data())
    Images = Images/np.amax(Images)
    Masks = Masks/np.amax(Masks)
    start_x =  int(Images.shape[0]/2-128)
    stop_x = int(Images.shape[0]/2+128)
    start_y = int(Images.shape[1]/2-128)
    stop_y = int(Images.shape[1]/2+128)
    Images = Images[start_x:stop_x,start_y:stop_y,:]
    Masks = Masks[start_x:stop_x,start_y:stop_y,:]
  
    locals()["image_extra_p%s" % number] = Images
    locals()["mask_extra_p%s" % number] = Masks
