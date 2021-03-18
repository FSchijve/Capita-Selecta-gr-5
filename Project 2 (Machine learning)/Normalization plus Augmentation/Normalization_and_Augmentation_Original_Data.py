import torch
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torchvision
import random
import difflib
import scipy.spatial
from glob import glob
import SimpleITK as sitk
import matplotlib.pyplot as plt
import gryds
import random
import imageio

#%%

# Normalization

def normalize_img(img): 
    # Enable when normalizing all values between 0 and 1:
    #img=img/np.amax(img)
    
    # Enable when normalizing mean 0 std 1:
    img = (img - np.mean(img))/np.std(img)
    
    # Cropping images into size 256 x 256
    start_x= int(img.shape[0]/2-128)
    stop_x = int(img.shape[0]/2+128)
    start_y= int(img.shape[1]/2-128)
    stop_y = int(img.shape[1]/2+128)
    img = img[start_x:stop_x,start_y:stop_y]
    return img

def normalize_mask(mask): 
    #mask[mask>1]=1
    start_x= int(mask.shape[0]/2-128)
    stop_x = int(mask.shape[0]/2+128)
    start_y= int(mask.shape[1]/2-128)
    stop_y = int(mask.shape[1]/2+128)
    mask  = mask[start_x:stop_x,start_y:stop_y]
    return mask

#%%
    
def Bspline(img, mask):
    # Define a random 3x3 B-spline grid for a 2D image:
    random_grid = np.random.rand(2, 3, 3)
    random_grid -= 0.5
    random_grid /= 5

    # Define a B-spline transformation object
    bspline = gryds.BSplineTransformation(random_grid)

    # Define an interpolator object for the image:
    interpolator_img = gryds.Interpolator(img)
    interpolator_mask = gryds.Interpolator(mask)
    
    # Transform the image using the B-spline transformation
    transformed_image = interpolator_img.transform(bspline)
    transformed_mask = interpolator_mask.transform(bspline)

    return transformed_image, transformed_mask

def Affine(img,mask):
    # Define a scaling transformation object
    angle = random.randrange(-1,2,2)*np.pi/(random.randint(6,16))
    center_point_x = 0.1*random.randint(3,7)
    center_point_y = 0.1*random.randint(3,7)
    affine = gryds.AffineTransformation(
    ndim=2,
    angles=[angle], # List of angles (for 3D transformations you need a list of 3 angles).
    center=[center_point_x, center_point_y])  # Center of rotation.
    
    # Define an interpolator object for the image:
    interpolator_img = gryds.Interpolator(img)
    interpolator_mask = gryds.Interpolator(mask)
    
    # Transform image and mask using Affine transformation
    transformed_image = interpolator_img.transform(affine)
    transformed_mask = interpolator_mask.transform(affine)
    return transformed_image, transformed_mask

def flip(img, mask): # Check if it properly works
    img = torch.from_numpy(img.copy())
    mask = torch.from_numpy(mask.copy())
    flipped_img = torchvision.transforms.functional.hflip(img = img) # change to .vflip for vertical flip
    flipped_mask = torchvision.transforms.functional.hflip(img = mask)
    flipped_img = flipped_img.cpu().detach().numpy()
    flipped_mask = flipped_mask.cpu().detach().numpy()
    return flipped_img, flipped_mask

def Bspline_and_Affine(img, mask):
    # Define a scaling transformation object
    angle = random.randrange(-1,2,2)*np.pi/(random.randint(6,16))
    center_point_x = 0.1*random.randint(3,7)
    center_point_y = 0.1*random.randint(3,7)
    affine = gryds.AffineTransformation(
    ndim=2,
    angles=[angle], # List of angles (for 3D transformations you need a list of 3 angles).
    center=[center_point_x, center_point_y])  # Center of rotation.
    
    # Define a random 3x3 B-spline grid for a 2D image:
    random_grid = np.random.rand(2, 3, 3)
    random_grid -= 0.5
    random_grid /= 5
    
    # Define a B-spline transformation object
    bspline = gryds.BSplineTransformation(random_grid)
    
    # Define an interpolator object for the image:
    interpolator_img = gryds.Interpolator(img)
    interpolator_mask = gryds.Interpolator(mask)
    
    # Transform the image using both transformations. The B-spline is applied to the
    # sampling grid first, and the affine transformation second. From the
    # perspective of the image itself, the order will seem reversed (!).
    transformed_image = interpolator_img.transform(bspline, affine)
    transformed_mask = interpolator_mask.transform(bspline, affine)
    return transformed_image, transformed_mask
    
def Bspline_and_Affine_flipped(img, mask):
    # Define a scaling transformation object
    angle = random.randrange(-1,2,2)*np.pi/(random.randint(6,16))
    center_point_x = 0.1*random.randint(3,7)
    center_point_y = 0.1*random.randint(3,7)
    affine = gryds.AffineTransformation(
    ndim=2,
    angles=[angle], # List of angles (for 3D transformations you need a list of 3 angles).
    center=[center_point_x, center_point_y])  # Center of rotation.
    
    # Define a random 3x3 B-spline grid for a 2D image:
    random_grid = np.random.rand(2, 3, 3)
    random_grid -= 0.5
    random_grid /= 5
    
    # Define a B-spline transformation object
    bspline = gryds.BSplineTransformation(random_grid)
    
    # Define an interpolator object for the image:
    interpolator_img = gryds.Interpolator(img)
    interpolator_mask = gryds.Interpolator(mask)
    
    # Transform the image using both transformations. The B-spline is applied to the
    # sampling grid first, and the affine transformation second. From the
    # perspective of the image itself, the order will seem reversed (!).
    transformed_image = interpolator_img.transform(bspline, affine)
    transformed_mask = interpolator_mask.transform(bspline, affine)
    
    img = torch.from_numpy(transformed_image.copy())
    mask = torch.from_numpy(transformed_mask.copy())
    flipped_img = torchvision.transforms.functional.hflip(img = img) # change to .vflip for vertical flip
    flipped_mask = torchvision.transforms.functional.hflip(img = mask)
    transformed_image = flipped_img.cpu().detach().numpy()
    transformed_mask = flipped_mask.cpu().detach().numpy()
    
    return transformed_image, transformed_mask
    
#%%

# Opening external dataset
data_path = r"C:\Users\s160518\Documents\8DM20 Project\TrainingData"


# number_list SHORT, this list should be used just to check the code for the first 3 patients
number_list = [102, 107]

# number_list COMPLETE, this list should be used in the end for all the patients
#number_list = [102, 107, 108, 109, 115, 116, 117, 119, 12, 125, 127, 128, 129, 133, 135]

number_of_slices = 86

i = -1
List_images = []
List_masks = []

# loop the patients
#for number in number_list:
for number in number_list:   
    i = i + 1
    img_path = os.path.join(data_path,f"p{number}\mr_bffe.mhd")
    mask_path  = os.path.join(data_path,f"p{number}\prostaat.mhd")
    
    List_img0 = number_of_slices * [0]
    List_img1 = number_of_slices * [0]
    List_img2 = number_of_slices * [0]
    List_img3 = number_of_slices * [0]
    List_img4 = number_of_slices * [0]
    List_img5 = number_of_slices * [0]
    
    List_mask0 = number_of_slices * [0]
    List_mask1 = number_of_slices * [0]
    List_mask2 = number_of_slices * [0]
    List_mask3 = number_of_slices * [0]
    List_mask4 = number_of_slices * [0]
    List_mask5 = number_of_slices * [0]
    # Loop the slices 
    #for slice in range(slice_list[i]):
    for slice in range(number_of_slices):
        mask = imageio.imread(mask_path)[slice,:,:]
        img  = imageio.imread(img_path)[slice,:,:]
        print ('Patient',number,'slice',slice)
        img = normalize_img(img)
        mask = normalize_mask(mask)
        
        # Original image
        List_img0[slice] = img
        List_mask0[slice] = mask
        
        # Bspline and Affine and flipped transformation
        img_bspline_affine_flipped, mask_bspline_affine_flipped = Bspline_and_Affine_flipped(img, mask)
        
        List_img1[slice] = img_bspline_affine_flipped
        List_mask1[slice] = mask_bspline_affine_flipped
        
        # Bspline and Affine and flipped transformation second time
        img_bspline_affine_flipped1, mask_bspline_affine_flipped1 = Bspline_and_Affine_flipped(img, mask)
        
        List_img2[slice] = img_bspline_affine_flipped1
        List_mask2[slice] = mask_bspline_affine_flipped1
        
        # Bspline and Affine transformation
        img_bspline_affine, mask_bspline_affine = Bspline_and_Affine(img, mask)
        
        List_img3[slice] = img_bspline_affine
        List_mask3[slice] = mask_bspline_affine
        
        # Bspline and Affine transformation second time
        img_bspline_affine1, mask_bspline_affine1 = Bspline_and_Affine(img, mask)
        
        List_img4[slice] = img_bspline_affine1
        List_mask4[slice] = mask_bspline_affine1
        
        # Bspline transformation
        img_bspline, mask_bspline = Bspline(img,mask)
        
        List_img5[slice] = img_bspline
        List_mask5[slice] = mask_bspline
        
    List_images.extend([List_img0, List_img1, List_img2, List_img3, List_img4, List_img5])
    List_masks.extend([List_mask0, List_mask1, List_mask2, List_mask3, List_mask4, List_mask5])


image_plot = List_images[5][32] # 5th created image, slice 32
mask_plot = List_masks[5][32]   

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image_plot, cmap='gray')
ax[0].set_title('Image')
ax[1].imshow(mask_plot,cmap='gray')
ax[1].set_title('Mask')

