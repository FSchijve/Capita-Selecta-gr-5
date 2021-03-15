# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 12:25:53 2021

@author: giuli
"""

#faluty script 

import os

data_path = (r'C:\Users\giuli\Desktop\Uni Utrecht\Capita Selecta\Project\TrainingData\Task05_Prostate')

#number_list = [00, 01, 02, 04, 06, 07, 10, 13, 14, 16, 17, 18, 20, 21, 24, 25, 28, 29, 31, 32, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47]

number_list = [00];
slice = 7; #slice to visualize 
x=256
y=256


for number in number_list:
    mask_path = os.path.join(data_path,f"imagesTr\prostate_{number}.nii.gz"); 
    img_path  = os.path.join(data_path,f"labelsTr\prostate_{number}.nii.gz");
    

# First, we import PyTorch and NumPy
import torch
import numpy as np
import os
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
from tqdm.notebook import tqdm # progressbar 

import nibabel as nib

class transforms(): 
  def normalize_img(self, img): 
    img=img/np.amax(img)
    start_x= int(img.shape[0]/2-128)
    stop_x = int(img.shape[0]/2+128)
    start_y= int(img.shape[1]/2-128)
    stop_y = int(img.shape[1]/2+128)
    img = img[:,start_x:stop_x,start_y:stop_y]
    return img

  def normalize_mask(self, mask): 
    mask[mask>1]=1
    start_x= int(mask.shape[0]/2-128)
    stop_x = int(mask.shape[0]/2+128)
    start_y= int(mask.shape[1]/2-128)
    stop_y = int(mask.shape[1]/2+128)
    mask  = mask[:,start_x:stop_x,start_y:stop_y]
    return mask
    
    #max_img = torch.max(img)
    #min_img = torch.min(img)
    #nom = (img - min_img) * (x_max - x_min)
    #denom = max_img - min_img
    #denom = denom + (denom == 0) 
    #return x_min + nom / denom 

  def rotate(self, img, mask, degrees): 
    """ Function to rotate both the image and mask with a random rotation in the same way.
    The degrees paramater has to be passed as a range e.g. (-18, 18).
    """
    angle = torchvision.transforms.RandomRotation.get_params(degrees)
    rotated_img = torchvision.transforms.functional.rotate(img, angle)
    rotated_mask = torchvision.transforms.functional.rotate(mask, angle)
    return rotated_img, rotated_mask

  def flip(self, img, mask): # Check if it properly works
    flipped_img = torchvision.transforms.functional.hflip(img = img) # change to .vflip for vertical flip
    flipped_mask = torchvision.transforms.functional.hflip(img = mask)
    return flipped_img, flipped_mask

  def scale(self, img, mask, range=0.2): # Check if it properly works
    """
    Function to scale both the image and the mask mask with a random range in the same way
    The range parameter is a float that will create a scaled image in the range of 1+- range
    has not yet been checked to see if it works
    """
    scale = random.randrange((1-range)*1000, (1+range)*1000)/1000
    scaled_img = torchvision.transforms.functional.affine(img=img, angle=0, translate=[0,0], shear=0, scale=scale)
    scaled_mask = torchvision.transforms.functional.affine(img=mask, angle=0, translate=[0,0], shear=0, scale=scale)
    return scaled_img, scaled_mask

  def shear(self, img, mask, degrees): # Check if it properly works.
    degree = np.random.randint(-degrees, degrees)
    sheared_img = torchvision.transforms.functional.affine(img = img, shear = [degree],
                                                         angle = 0, translate = [0,0], scale = 1)
    sheared_mask = torchvision.transforms.functional.affine(img = mask, shear = [degree],
                                                         angle = 0, translate = [0,0], scale = 1)
    return sheared_img, sheared_mask


class maskDataset(torch.utils.data.Dataset):
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
    
  def __init__(self, normalize_img = False, normalize_mask = False, rotate = (False, 0) , flip = False, scale = False, shear = (False, 0)):
    
    self.x=x
    self.y=y
    self.slices = slice 
    self.datafolder = glob(data_path)
    #print("self.datafolder is: ", self.datafolder)
    #print("patientnr", patientnr)

    # Initializations for data augmentation
    self.transforms = transforms()

    # I'd suggest passing any extra parameters necessary for the transformation along with the variable as a tuple.
    # Then unpack the tuple here and use it later, when applying the augmentation. This way those parameters are not fixed inside the class.
    self.normalize_img  = normalize_img
    self.normalize_mask = normalize_mask
    self.rotate, self.rotation_angle = rotate
    self.flip  = flip
    self.scale = scale
    self.shear, self.shear_angle = shear

  def __len__(self): # the length is the number of patients scanned at the institute. Every patient is a subfolder in the institute folder
    return len(self.datafolder) * self.slices


  # This is a helper function to avoid repeating the same SimpleITK function calls to load the images
  # It loads the Nifti files, gets a correctly spaced NumPy array, and creates a tensor
  def read_image(self, path):
    img = sitk.ReadImage(path)
    img_as_numpy = sitk.GetArrayFromImage(img).astype('float') # the default type is uint16, which trips up PyTorch so we convert to float
    img_as_tensor = torch.from_numpy(img_as_numpy)
    return img_as_tensor

  def __getitem__(self, i): # return the ith sample of the dataset, note that 0 <= i < len(dataset)
    # A slice is considered a sample.
         
    # Actually load the Nifti files and create PyTorch tensors
    #nib.load().get_data()
    mask = self.read_image(mask_path)
    img  = self.read_image(img_path)
  
    #_, x, y = mask.size()
    train_tensor = torch.zeros((1, self.x, self.y)) # Use only one to avoid error shown.
    target_tensor = torch.zeros((1, x, y))
    
    # slice_index = i % self.slices
    train_tensor[0, ...] = img[slice, ...]
    target_tensor[0, ...]= mask[slice, ...]

    # Apply normalization
    if self.normalize_img:
      train_tensor = self.transforms.normalize_img(train_tensor)
    
    if self.normalize_mask:
      train_tensor = self.transforms.normalize_mask(train_tensor)
    
    # Apply data augmentation
    if self.rotate:
      train_tensor, target_tensor = self.transforms.rotate(train_tensor, target_tensor, self.rotation_angle)
    
    if self.flip:
      train_tensor, target_tensor = self.transforms.flip(train_tensor, target_tensor)
    
    if self.scale:
      train_tensor, target_tensor = self.transforms.scale(train_tensor, target_tensor)
    
    if self.shear:
      train_tensor, target_tensor = self.transforms.shear(train_tensor, target_tensor, self.shear_angle)

    # Return the samples as PyTorch tensors
    return train_tensor, target_tensor


dataset = maskDataset() # Note this dataset is now already normalized.
len(dataset)

#Take a look at what the data looks like
train, target = dataset[slice]
train.size()


# create the figure
showFig = 'yes'

if showFig=='yes':
  f, ax = plt.subplots(1, 3, figsize=(15, 15))

  # turn off axis to remove ticks and such
  [a.axis('off') for a in ax]

  # Here we plot it at the actual subplot we want. We set the colormap to gray (feel free to experiment)
  img_plot = ax[0].imshow(train[0, :, :], cmap='gray') # Was one but we only working with flair for now.
  mask_plot = ax[1].imshow(target[0, :, :], cmap='gray')

  # Add titles and colorbar
  ax[0].set_title('Image')
  ax[1].set_title('Previously provided mask')

  f.colorbar(img_plot, ax=ax[0], shrink=0.25)
  f.colorbar(mask_plot, ax=ax[1], shrink=0.25)