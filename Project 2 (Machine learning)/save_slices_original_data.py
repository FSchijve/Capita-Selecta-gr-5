import torch
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torchvision
import random
import nibabel as nib
import gryds
import cv2
import matplotlib.pyplot as plt
import imageio

#%%
# Standard variables

# Opening external dataset
data_path = r"C:\Users\s160518\Documents\8DM20 Project\TrainingData"
processed_data_path = r"C:\Users\s160518\Documents\CSMIA Segmentation\preprocesseddata"


# Patient list COMPLETE, this list should be used in the end for all the patients
patient_list = [102, 107, 108, 109, 115, 116, 117, 119, 12, 125, 127, 128, 129, 133, 135]

#%%
# Variables to change

# Patient list SHORT, this list should be used just to check the code for the first few patients
patient_list = patient_list[:1]

image_side = 128

#%%

# Normalization

def normalize_img(img): 
    # Enable when normalizing all values between 0 and 1:
    #img=img/np.amax(img)
    
    # Enable when normalizing mean 0 std 1:
    img = (img - np.mean(img))/np.std(img)
    
    # resizing images to 128 x 128
    img = cv2.resize(img, (128,128), interpolation = cv2.INTER_CUBIC)
    return img

def normalize_mask(mask): 
    # resizing masks to 128 x 128
    mask = cv2.resize(mask, (128,128), interpolation = cv2.INTER_NEAREST)
    return mask

def normalize(img, mask):
    return normalize_img(img), normalize_mask(mask)

#%%

# Transforms
    
def Bspline(img, mask):
    # Define a random 3x3 B-spline grid for a 2D image:
    random_grid = np.random.rand(2, 3, 3)
    random_grid -= 0.5
    random_grid /= 10

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
    center_point_x = 0.1*random.randint(4,6)
    center_point_y = 0.1*random.randint(4,6)
    affine = gryds.AffineTransformation(
            ndim=2,
            angles=[random.randrange(-1,2,2)*(np.pi/(random.randint(50,60)))], # List of angles (for 3D transformations you need a list of 3 angles).
            center=[center_point_x, center_point_y]  # Center of rotation.
            )


    # Define an interpolator object for the image:
    interpolator_img = gryds.Interpolator(img)
    interpolator_mask = gryds.Interpolator(mask)

    # Transform the image using Affine
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
    center_point_x = 0.1*random.randint(4,6)
    center_point_y = 0.1*random.randint(4,6)
    affine = gryds.AffineTransformation(
            ndim=2,
            angles=[random.randrange(-1,2,2)*(np.pi/(random.randint(50, 60)))], # List of angles (for 3D transformations you need a list of 3 angles).
            center=[center_point_x, center_point_y]  # Center of rotation.
            )
    
    # Define a random 3x3 B-spline grid for a 2D image:
    random_grid = np.random.rand(2, 3, 3)
    random_grid -= 0.5
    random_grid /= 10
    
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
    center_point_x = 0.1*random.randint(4,6)
    center_point_y = 0.1*random.randint(4,6)
    affine = gryds.AffineTransformation(
            ndim=2,
            angles=[random.randrange(-1,2,2)*(np.pi/(random.randint(50,60)))], # List of angles (for 3D transformations you need a list of 3 angles).
            center=[center_point_x, center_point_y]  # Center of rotation.
            )
    
    # Define a random 3x3 B-spline grid for a 2D image:
    random_grid = np.random.rand(2, 3, 3)
    random_grid -= 0.5
    random_grid /= 10
    
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

# Dataset classes

class Dataset():
    # Single dataset with slices
    def __init__(self, image_paths = [], filename = None):
        self.image_paths = image_paths.copy()
        if filename != None:
            self.read(filename)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        return np.load(self.image_paths[idx])
    
    def __setitem__(self, idx, new_path):
        self.image_paths[idx] = new_path
            
    def getpath(self, idx):
    # Get the path of a certain slice
        return self.image_paths[idx]
    
    def addimage(self, image_path):
    # Add a single slice to the dataset
        self.image_paths.append(image_path)

    def addimages(self, image_paths):
    # Add a list of slices to the dataset
        for image_path in image_paths:
            self.image_paths.append(image_path)

    def adddataset(self, dataset):
    # Add a dataset to the dataset
        for i in range(len(dataset)):
            self.image_paths.append(dataset.getpath(i))

    def adddatasets(self, datasets):
    # Add a list of datasets to the dataset
        for i in range(len(datasets)):
            for j in range(len(datasets[i])):
                self.image_paths.append(datasets[i].getpath(j))
    
    def shuffle(self, seed):
    # Shuffle the dataset
        random.Random(seed).shuffle(self.image_paths)
        
    def write(self, filename):
    # Write the dataset to a file
        with open(filename,'w') as f:
            for image_path in self.image_paths:
                f.writelines(image_path+'\n')
        
    def read(self, filename):
    # Add slices from a dataset file
        with open(filename,'r') as f:
            lines = f.readlines()

        for line in lines:
            self.image_paths.append(line[:-1])            
        
class XY_dataset():
    # Combined dataset (x = images, y = masks)
    def __init__(self, x_set, y_set, batch_size = 1, end_evaluation = False, verbose = False):
        if len(x_set) != len(y_set):
            raise Exception("Length of x_set is not the same as length of y_set.")
        self.x_set, self.y_set = x_set, y_set
        self.n = 0
        self.max = len(x_set)
        self.batch_size = batch_size
        self.end_evaluation = end_evaluation
        self.verbose = verbose
        
    def __iter__(self):
        return self
    
    def __next__(self):
        # Return next batch        
        
        x_array, y_array = [], []
        for _ in range(0,self.batch_size):
            if self.verbose: print(f"Read slice {self.n}")
                    
            x_array.append(self.x_set[self.n])
            y_array.append(self.y_set[self.n])
    
            self.n += 1
            if self.n >= self.max:
                self.n = 0
                if self.end_evaluation:
                    raise StopIteration

        return (np.array(x_array),np.array(y_array))

    def __len__(self):
        return len(self.x_set)
    
    def set_end_evaluation(self, bool):
        self.end_evaluation = bool


#%%
    
def process_image(img, mask, dataset_img, dataset_mask, image_nr, function = None):
    # Transform, reshape, save and add to dataset

    if function == None:
        img_processed, mask_processed = img, mask
    else:
        img_processed, mask_processed = function(img, mask)

    img_processed = np.reshape(img_processed, (image_side,image_side,1))    
    np.save(os.path.join(processed_data_path,f"slice_{image_nr}.npy"),img_processed)
    dataset_img.addimage(os.path.join(processed_data_path,f"slice_{image_nr}.npy"))
    image_nr += 1

    mask_processed = np.reshape(mask_processed, (image_side,image_side,1))    
    np.save(os.path.join(processed_data_path,f"slice_{image_nr}.npy"),mask_processed)
    dataset_mask.addimage(os.path.join(processed_data_path,f"slice_{image_nr}.npy"))
    image_nr += 1    

    return image_nr

#%%

# Do augmentations and save them

if __name__ == '__main__':
    if not os.path.isdir(processed_data_path):
        os.mkdir(processed_data_path)

    image_nr = 0
    train_images = Dataset()
    train_masks = Dataset()
    val_images = Dataset()
    val_masks = Dataset()
    
    # loop over the patients
    for patient_nr in patient_list:   
        img_path = os.path.join(data_path,f"p{patient_nr}\mr_bffe.mhd")
        mask_path  = os.path.join(data_path,f"p{patient_nr}\prostaat.mhd")
        
        orig_img, orig_mask = Dataset(), Dataset()
        baf0_img, baf0_mask = Dataset(), Dataset()
        baf1_img, baf1_mask = Dataset(), Dataset()
        ba0_img, ba0_mask = Dataset(), Dataset()
        ba1_img, ba1_mask = Dataset(), Dataset()
        b_img, b_mask = Dataset(), Dataset()
    
        nr_slices = imageio.imread(mask_path).shape[0]
    
        # Loop the slices 
        for slice in range(nr_slices):
            mask = imageio.imread(mask_path)[slice,:,:]
            img  = imageio.imread(img_path)[slice,:,:] 
            print ('Patient',patient_nr,'slice',slice)

            # Process images in different ways
            img, mask = normalize(img, mask)
            image_nr = process_image(img, mask, orig_img, orig_mask, image_nr)
            image_nr = process_image(img, mask, baf0_img, baf0_mask, image_nr, Bspline_and_Affine_flipped)
            image_nr = process_image(img, mask, baf1_img, baf1_mask, image_nr, Bspline_and_Affine_flipped)
            image_nr = process_image(img, mask, ba0_img, ba0_mask, image_nr, Bspline_and_Affine)
            image_nr = process_image(img, mask, ba1_img, ba1_mask, image_nr, Bspline_and_Affine)
            image_nr = process_image(img, mask, b_img, b_mask, image_nr, Bspline)

        train_images.adddatasets([baf0_img,baf1_img,ba0_img,ba1_img,b_img])
        train_masks.adddatasets([baf0_mask,baf1_mask,ba0_mask,ba1_mask,b_mask])
        val_images.adddataset(orig_img)
        val_masks.adddataset(orig_mask)    
    
    train_images.write(os.path.join(processed_data_path,"train_images.txt"))
    train_masks.write(os.path.join(processed_data_path,"train_masks.txt"))
    val_images.write(os.path.join(processed_data_path,"val_images.txt"))
    val_masks.write(os.path.join(processed_data_path,"val_masks.txt"))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img, cmap='gray')
ax[0].set_title('Image')
ax[1].imshow(mask,cmap='gray')
ax[1].set_title('Mask')
