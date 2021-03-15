# First, we import PyTorch and NumPy
import torch
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# For the augmentations
import torchvision
import random
# These two extra for evaluation.
import difflib
import scipy.spatial
# We import glob to find everything that matches a pattern
from glob import glob
# We install and import SimpleITK for image loading
# pip is the package installer for python
import SimpleITK as sitk
# To show data, we import matplotlib
import matplotlib.pyplot as plt
#from tqdm.notebook import tqdm # progressbar 

import nibabel as nib

#%%

# rigid transformation functions

def normalize_img(img): 
    img=img/np.amax(img)
    start_x= int(img.shape[0]/2-128)
    stop_x = int(img.shape[0]/2+128)
    start_y= int(img.shape[1]/2-128)
    stop_y = int(img.shape[1]/2+128)
    img = img[start_x:stop_x,start_y:stop_y]
    return img

def normalize_mask(mask): 
    mask[mask>1]=1
    start_x= int(mask.shape[0]/2-128)
    stop_x = int(mask.shape[0]/2+128)
    start_y= int(mask.shape[1]/2-128)
    stop_y = int(mask.shape[1]/2+128)
    mask  = mask[start_x:stop_x,start_y:stop_y]
    return mask
    

    #max_img = torch.max(img)
    #min_img = torch.min(img)
    #nom = (img - min_img) * (x_max - x_min)
    #denom = max_img - min_img
    #denom = denom + (denom == 0) 
    #return x_min + nom / denom 

def rotate(img, mask, degrees): 
    """ Function to rotate both the image and mask with a random rotation in the same way.
    The degrees paramater has to be passed as a range e.g. (-18, 18).
    """
    img = torch.from_numpy(img.copy())
    mask = torch.from_numpy(mask.copy())
    angle = torchvision.transforms.RandomRotation.get_params(degrees)
    rotated_img = torchvision.transforms.functional.rotate(img, angle)
    rotated_mask = torchvision.transforms.functional.rotate(mask, angle)
    rotated_img = rotated_img.cpu().detach().numpy()
    rotated_mask = rotated_mask.cpu().detach().numpy()
    return img, mask

def flip( img, mask): # Check if it properly works
    img = torch.from_numpy(img.copy())
    mask = torch.from_numpy(mask.copy())
    flipped_img = torchvision.transforms.functional.hflip(img = img) # change to .vflip for vertical flip
    flipped_mask = torchvision.transforms.functional.hflip(img = mask)
    flipped_img = flipped_img.cpu().detach().numpy()
    flipped_mask = flipped_mask.cpu().detach().numpy()
    return flipped_img, flipped_mask

def scale( img, mask, range=0.2): # Check if it properly works
    """
    Function to scale both the image and the mask mask with a random range in the same way
    The range parameter is a float that will create a scaled image in the range of 1+- range
    has not yet been checked to see if it works
    """
    img = torch.from_numpy(img.copy())
    mask = torch.from_numpy(mask.copy())
    scale = random.randrange((1-range)*1000, (1+range)*1000)/1000
    scaled_img = torchvision.transforms.functional.affine(img=img, angle=0, translate=[0,0], shear=0, scale=scale)
    scaled_mask = torchvision.transforms.functional.affine(img=mask, angle=0, translate=[0,0], shear=0, scale=scale)
    scaled_img = scaled_img.cpu().detach().numpy()
    scaled_mask = scaled_mask.cpu().detach().numpy()
    return scaled_img, scaled_mask

def shear( img, mask, degrees): # Check if it properly works.
    img = torch.from_numpy(img.copy())
    mask = torch.from_numpy(mask.copy())
    degree = np.random.randint(-degrees, degrees)
    sheared_img = torchvision.transforms.functional.affine(img = img, shear = [degree],
                                                         angle = 0, translate = [0,0], scale = 1)
    sheared_mask = torchvision.transforms.functional.affine(img = mask, shear = [degree],
                                                         angle = 0, translate = [0,0], scale = 1)
    sheared_img = sheared_img.cpu().detach().numpy()
    sheared_mask = sheared_mask.cpu().detach().numpy()
    return sheared_img, sheared_mask


#%%

  # Datasets in Pytorch are classes of the torch.utils.data.Dataset type
  # They __must__ have at least three methods:
  # - __init__ -> Initialize the dataset, the place where you can pass parameters to it
  # - __len__ -> How many samples does your dataset represent?
  # - __getitem__ -> A function which takes a parameter i, and returns the ith sample of the dataset

  # Note that this DOES NOT perform
  # - Batching
  # - Asynchronous dataloading (for speed)
  # - Merge different datasets on the fly 
  # - shuffling the data
  # More examples like these are solved with "higher-order" methods

  # but it __might__ do:
  # - data augmentation of one sample
  # - data normalization of one sample
  # - performing on-the-fly data generation
  # - hides the nitty-gritty details of dealing with files
    


  # This is a helper function to avoid repeating the same SimpleITK function calls to load the images
  # It loads the Nifti files, gets a correctly spaced NumPy array, and creates a tensor
def read_image( path):
    img = sitk.ReadImage(path)
    img_as_numpy = sitk.GetArrayFromImage(img).astype('float') # the default type is uint16, which trips up PyTorch so we convert to float
    img_as_tensor = torch.from_numpy(img_as_numpy)
    return img_as_tensor

 
#%%

# Opening external dataset
#data_path = (r'C:\Users\darja\Documents\TuE\elastix-5.0.1-win64\TrainingData\Task05_Prostate')
data_path = r"C:\Users\s160518\Documents\CSMIA Segmentation\Task05_Prostate"
number_list = [0, 1, 2, 4, 6, 7, 10, 13, 14, 16, 17, 18, 20, 21, 24, 25, 28, 29, 31, 32, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47]

   
slice_list = [15, 20, 24, 15, 20, 20, 20, 20, 20, 20, 20, 18, 20, 20, 20, 19, 11, 15, 20, 20, 15, 20, 15, 20, 15, 20, 18, 22, 20, 20, 20, 20]
i = -1

# loop the patients
#for number in number_list:
for number in range(1):   
    i = i + 1
    mask_path = os.path.join(data_path,f"labelsTr\prostate_{number}.nii.gz"); 
    img_path  = os.path.join(data_path,f"imagesTr\prostate_{number}.nii.gz")
   
    # Loop the slices 
    #for slice in range(slice_list[i]):
    for slice in range(7,8):
        mask = np.rot90(nib.load(mask_path).get_data()[:,:,slice])
        img  = np.rot90(nib.load(img_path).get_data()[:,:,slice,0]) 
        print (i)
        img = normalize_img(img)
        mask = normalize_mask(mask)
        img_flipped,_=flip(img, mask)
        #Take a look at what the data looks like
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Original')
ax[1].imshow(img_flipped,cmap='gray')
ax[1].set_title('Flipped')

    
# create the figure
        # showFig = 'yes'
    
        #   if showFig=='yes':
        #   f, ax = plt.subplots(1, 3, figsize=(15, 15))
        
        #   # turn off axis to remove ticks and such
        #   [a.axis('off') for a in ax]
        
        #   # Here we plot it at the actual subplot we want. We set the colormap to gray (feel free to experiment)
        #   img_plot  = ax[0].imshow(img, cmap='gray') # Was one but we only working with flair for now.
        #   mask_plot = ax[1].imshow(mask, cmap='gray')
        
        #   # Add titles and colorbar
        #   ax[0].set_title('Image')
        #   ax[1].set_title('Previously provided mask')
        
        #   f.colorbar(img_plot, ax=ax[0], shrink=0.25)
        #   f.colorbar(mask_plot, ax=ax[1], shrink=0.25)