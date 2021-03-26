import torch
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torchvision
import random
import nibabel as nib
import gryds
import cv2
import imageio
from sklearn.utils import shuffle
import math

#%%
# Variables to change

patient_list_length = 20
image_side = 128
dataset_type = "prostate_extra"

#%%
# Standard variables

# Opening external dataset
if dataset_type == "prostate_extra":
    processed_data_path = r"C:\Users\Dell\Documents\Medical_Imaging\CSMI_TUE\preprocessed_data_prostate_extra"
    data_path = r"C:\Users\Dell\Documents\Medical_Imaging\CSMI_TUE\data\new"

    patient_list = ['00', '01', '02', '04', '06', '07', 10, 13, 14, 16, 17, 18, 20, 21, 24, 25, 28, 29, 31, 32, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47]

    target_img_paths = [os.path.join(data_path,f"labelsTr\prostate_{patient_nr}.nii.gz") for patient_nr in patient_list]
    input_img_paths  = [os.path.join(data_path,f"imagesTr\prostate_{patient_nr}.nii.gz") for patient_nr in patient_list]

elif dataset_type == "heart_extra":
    processed_data_path = r"C:\Users\Dell\Documents\Medical_Imaging\CSMI_TUE\preprocessed_data_heart_extra"
    data_path = r"C:\Users\Dell\Documents\Medical_Imaging\CSMI_TUE\data\Task02_Heart"

    patient_list = ['03', '04', '05', '07', '09', 10, 11, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 29, 30]

    target_img_paths = [os.path.join(data_path,f"labelsTr\la_0{patient_nr}.nii.gz") for patient_nr in patient_list]
    input_img_paths  = [os.path.join(data_path,f"imagesTr\la_0{patient_nr}.nii.gz") for patient_nr in patient_list]

elif dataset_type == "catsdogs":
    processed_data_path = r"C:\Users\Dell\Documents\Medical_Imaging\CSMI_TUE\preprocessed_data_catsdogs"
    input_dir = r"C:/Users/Dell/Documents/Medical_Imaging/CSMI_TUE/data/catsdogs/images/"
    target_dir = r"C:/Users/Dell/Documents/Medical_Imaging/CSMI_TUE/data/catsdogs/annotations/trimaps/"

    # Making list of paths for all images and masks
    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".jpg")])
    target_img_paths = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".png") and not fname.startswith(".")])

    # Some images gave errors (idk why), so I removed them manually
    input_img_paths.remove(os.path.join(input_dir,'Abyssinian_34.jpg'))
    input_img_paths.remove(os.path.join(input_dir,'Egyptian_Mau_139.jpg'))
    input_img_paths.remove(os.path.join(input_dir,'Egyptian_Mau_145.jpg'))
    input_img_paths.remove(os.path.join(input_dir,'Egyptian_Mau_167.jpg'))
    input_img_paths.remove(os.path.join(input_dir,'Egyptian_Mau_177.jpg'))
    input_img_paths.remove(os.path.join(input_dir,'Egyptian_Mau_191.jpg'))
    target_img_paths.remove(os.path.join(target_dir,'Abyssinian_34.png'))
    target_img_paths.remove(os.path.join(target_dir,'Egyptian_Mau_139.png'))
    target_img_paths.remove(os.path.join(target_dir,'Egyptian_Mau_145.png'))
    target_img_paths.remove(os.path.join(target_dir,'Egyptian_Mau_167.png'))
    target_img_paths.remove(os.path.join(target_dir,'Egyptian_Mau_177.png'))
    target_img_paths.remove(os.path.join(target_dir,'Egyptian_Mau_191.png'))

elif dataset_type == "original":
    processed_data_path = r"C:\Users\Dell\Documents\Medical_Imaging\CSMI_TUE\preprocessed_data_original"
    data_path = r"C:\Users\Dell\Documents\Medical_Imaging\CSMI_TUE\data"
    
    patient_list = [102, 107, 108, 109, 115, 116, 117, 119, 12, 125, 127, 128, 129, 133, 135]

    target_img_paths = [os.path.join(data_path,f"p{patient_nr}\prostaat.mhd") for patient_nr in patient_list]
    input_img_paths  = [os.path.join(data_path,f"p{patient_nr}\mr_bffe.mhd") for patient_nr in patient_list]

else:
    raise Exception(f"dataset_type {dataset_type} not recognized!")
    
# Patient list SHORT, this list should be used just to check the code for the first few patients
target_img_paths = target_img_paths[:patient_list_length]
input_img_paths = input_img_paths[:patient_list_length]

#%%

# Normalization

def normalize_img(img, dataset_type):
    # Enable when normalizing all values between 0 and 1:
    # img=img/np.amax(img)
    
    # Enable when normalizing mean 0 std 1:
    img = (img - np.mean(img))/np.std(img)
    
    # resizing images to image_side x image_side
    img = cv2.resize(img, (image_side,image_side), interpolation = cv2.INTER_CUBIC)
    return img
    
def normalize_mask(mask, dataset_type):
    if dataset_type in ["prostate_extra", "heart_extra", "original"]:
        mask[mask>1]=1
        # resizing masks to image_side x image_side
        mask = cv2.resize(mask, (image_side, image_side), interpolation = cv2.INTER_NEAREST)
        return mask
    
    elif dataset_type == "catsdogs":
        # resizing masks to image_side x image_side
        mask = cv2.resize(mask, (image_side,image_side), interpolation = cv2.INTER_NEAREST)
        max_value = np.amax(mask)
        min_value = np.amin(mask)
        for i in range(image_side):
            for j in range(image_side):
                if mask[i,j]==max_value:
                    mask[i,j]=1
                if mask[i,j]==min_value:
                    mask[i,j]=1
                else:
                    mask[i,j]=0
        return mask
        

def normalize(img, mask, dataset_type):
    return normalize_img(img, dataset_type), normalize_mask(mask, dataset_type)

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
    def __init__(self, image_side = None, filename = None):
        self.image_paths = [].copy()
        self.image_side = image_side
        if filename != None:
            self.read(filename)
        else:
            if image_side == None:
                raise Exception("Image side is not given!")
        
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
            self.addimage(image_path)

    def adddataset(self, dataset):
    # Add a dataset to the dataset
        for i in range(len(dataset)):
            self.addimage(dataset.getpath(i))

    def adddatasets(self, datasets):
    # Add a list of datasets to the dataset
        for i in range(len(datasets)):
            self.adddataset(datasets[i])
    
    def shuffle(self, seed):
    # Shuffle the dataset
        random.Random(seed).shuffle(self.image_paths)
        
    def write(self, filename):
    # Write the dataset to a file
        with open(filename,'w') as f:
            f.writelines(f"image_side = {self.image_side}\n")
            for image_path in self.image_paths:
                f.writelines(image_path+'\n')
        
    def read(self, filename):
    # Add slices from a dataset file
        self.image_paths = [].copy()
        self.image_side = 0

        with open(filename,'r') as f:
            lines = f.readlines()

        for line in lines:
            if line[:13] == "image_side = ":
                self.image_side = int(line[13:-1])
            else:
                self.image_paths.append(line[:-1])
        
        if self.image_side == 0:
            raise Exception("Old dataset version is used. Please create augmented slices again with this updated version of save_slices.")
        
class XY_dataset():
    # Combined dataset (x = images, y = masks)
    def __init__(self, x_set, y_set, batch_size = 1, end_evaluation = False, verbose = False):
        # Check if x_set and y_set have same length
        if len(x_set) != len(y_set):
            raise Exception("Length of x_set is not the same as length of y_set.")

        # Add datasets to object            
        self.x_set = x_set
        self.y_set = y_set
        
        # Add batchsize to object and check batch_size
        self.batch_size = batch_size
        if batch_size > len(x_set): raise Exception("Batch size must be smaller then dataset size.")

        # Determine number of batches
        self.number_of_batches = math.floor(len(x_set)/batch_size)

        # Locations of the empty and nonempty slices
        self.indices_empty, self.indices_nonempty = [], []
        for i in range(len(y_set)):
            if np.array([y_set[i]]).sum() == 0: # If slice is empty
                self.indices_empty.append(i)
            else: # If slice is not empty
                self.indices_nonempty.append(i)

        # Determine which fraction is empty and which fraction is not
        fraction_empty = len(self.indices_empty)/len(x_set)
        fraction_nonempty = 1-fraction_empty
        
        # Determine the batch sizes of the empty and nonempty slices
        self.batch_size_empty = [math.floor(fraction_empty*batch_size)]*self.number_of_batches
        self.batch_size_nonempty = [math.floor(fraction_nonempty*batch_size)]*self.number_of_batches
        #TODO CHECK IF BATCHSIZE IS LARGE ENOUGH (HAS NONEMPTY SLICES)
        # If the batch size is 1 less then we want
        if self.batch_size_empty[0]+self.batch_size_nonempty[0] != self.batch_size:
            # If the batch size is even less then that, something probably went wrong.
            if self.batch_size_empty[0]+self.batch_size_nonempty[0]+1 != self.batch_size:
                raise Exception("empty-nonempty split failed - code should be rewritten! (Call Aart)")

            # How many exra empty and nonempty slices we want
            n_empty_extra = round(fraction_empty*self.number_of_batches)
            n_nonempty_extra = self.number_of_batches - n_empty_extra
            if n_empty_extra > len(self.indices_empty)-sum(self.batch_size_empty):
                n_empty_extra = len(self.indices_empty)-sum(self.batch_size_empty)
                n_nonempty_extra = self.number_of_batches - n_empty_extra
            if n_nonempty_extra > len(self.indices_nonempty)-sum(self.batch_size_nonempty):
                n_nonempty_extra = len(self.indices_nonempty)-sum(self.batch_size_nonempty)
                n_empty_extra = self.number_of_batches - n_nonempty_extra

            # Create lists to add
            extra_empty = [1]*n_empty_extra + [0]*n_nonempty_extra
            extra_nonempty = [0]*n_empty_extra + [1]*n_nonempty_extra
            
            # Shuffle lists
            extra_empty, extra_nonempty = shuffle(extra_empty, extra_nonempty)
            
            # Add slices to batches
            for i in range(len(extra_empty)):
                self.batch_size_empty[i] += extra_empty[i]
                self.batch_size_nonempty[i] += extra_nonempty[i]
                            
        # Set trackers to 0
        self.batch_nr = 0
        self.empty_nr = 0
        self.nonempty_nr = 0

        # Other variables
        self.end_evaluation = end_evaluation
        self.last_step = False
        self.verbose = verbose
        if y_set.image_side != x_set.image_side:
            raise Exception("images in the x_set and y_set are not of the same size!")
        self.image_side = x_set.image_side
        
    def __len__(self):
        return len(self.x_set)

    def __iter__(self):
        return self
    
    def __next__(self):
    # Return next batch
        
        # To fix error:
        if self.end_evaluation and self.last_step:
            raise StopIteration

        # After one epoch
        if self.batch_nr == self.number_of_batches:
            # Stop end evaluation after this step
            if self.end_evaluation:
                self.last_step = True
            # And go to next epoch
            self.batch_nr = 0

        # Arrays with slices
        x_array, y_array = [], []
        
        # Add empty slices
        if self.verbose: print(f"\nRead {self.batch_size_empty[self.batch_nr]} empty slices")
        for _ in range(self.batch_size_empty[self.batch_nr]):
           # Repeat from start
            if self.empty_nr == len(self.indices_empty):
                self.empty_nr = 0

            if self.verbose: print(f"Read empty slice {self.empty_nr}")

            # Add slice
            x_array.append(self.x_set[self.indices_empty[self.empty_nr]])
            y_array.append(self.y_set[self.indices_empty[self.empty_nr]])
    
            self.empty_nr += 1
                        
        # Add nonempty slices
        if self.verbose: print(f"Read {self.batch_size_nonempty[self.batch_nr]} nonempty slices")
        for _ in range(self.batch_size_nonempty[self.batch_nr]):
            # Repeat from start
            if self.nonempty_nr == len(self.indices_nonempty):
                self.nonempty_nr = 0

            if self.verbose: print(f"Read nonempty slice {self.nonempty_nr}")

            # Add slice                    
            x_array.append(self.x_set[self.indices_nonempty[self.nonempty_nr]])
            y_array.append(self.y_set[self.indices_nonempty[self.nonempty_nr]])
    
            self.nonempty_nr += 1

        self.batch_nr += 1

        # Shuffle empty and nonempty slices
        x_array, y_array = shuffle(x_array, y_array)

        return (np.array(x_array),np.array(y_array))
    
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
    images = Dataset(image_side)
    masks = Dataset(image_side)
    
    # loop over the patients
    for i, paths in enumerate(zip(target_img_paths, input_img_paths)):
        mask_path, img_path = paths

        orig_img, orig_mask = Dataset(image_side), Dataset(image_side)
        if dataset_type in ["prostate_extra", "heart_extra", "original"]:
            baf0_img, baf0_mask = Dataset(image_side), Dataset(image_side)
            baf1_img, baf1_mask = Dataset(image_side), Dataset(image_side)
            ba0_img, ba0_mask = Dataset(image_side), Dataset(image_side)
            ba1_img, ba1_mask = Dataset(image_side), Dataset(image_side)
            b_img, b_mask = Dataset(image_side), Dataset(image_side)
    
        if dataset_type in ["prostate_extra", "heart_extra"]:
            nr_slices = nib.load(mask_path).get_fdata().shape[2]
        elif dataset_type == "original":
            nr_slices = imageio.imread(mask_path).shape[0]
        elif dataset_type == "catsdogs":
            print('cat/dog',i)
            nr_slices = 1
        
        # Loop the slices 
        for slice in range(nr_slices):
            if dataset_type == "prostate_extra":
                mask = np.rot90(nib.load(mask_path).get_fdata()[:,:,slice])
                img  = np.rot90(nib.load(img_path).get_fdata()[:,:,slice,0])
                print ('Patient step',i,'slice',slice)

            if dataset_type == "heart_extra":
                mask = np.rot90(nib.load(mask_path).get_fdata()[:,:,slice])
                img  = np.rot90(nib.load(img_path).get_fdata()[:,:,slice])
                print ('Patient step',i,'slice',slice)

            elif dataset_type == "original":
                mask = imageio.imread(mask_path)[slice,:,:]
                img  = imageio.imread(img_path)[slice,:,:]                

            elif dataset_type == "catsdogs":
                mask = imageio.imread(mask_path)
                img = imageio.imread(img_path)
                
                # some images are different shapes (:,:,3) and some only (:,:)
                if len(img.shape)>2:
                    img = img[:,:,0]

            # Process images in different ways
            img, mask = normalize(img, mask, dataset_type)
            image_nr = process_image(img, mask, orig_img, orig_mask, image_nr)
            if dataset_type in ["prostate_extra", "heart_extra", "original"]:
                image_nr = process_image(img, mask, baf0_img, baf0_mask, image_nr, Bspline_and_Affine_flipped)
                image_nr = process_image(img, mask, baf1_img, baf1_mask, image_nr, Bspline_and_Affine_flipped)
                image_nr = process_image(img, mask, ba0_img, ba0_mask, image_nr, Bspline_and_Affine)
                image_nr = process_image(img, mask, ba1_img, ba1_mask, image_nr, Bspline_and_Affine)
                image_nr = process_image(img, mask, b_img, b_mask, image_nr, Bspline)

        images.adddataset(orig_img)
        masks.adddataset(orig_mask)
        if dataset_type in ["prostate_extra", "heart_extra", "original"]:
            images.adddatasets([baf0_img,baf1_img,ba0_img,ba1_img,b_img])
            masks.adddatasets([baf0_mask,baf1_mask,ba0_mask,ba1_mask,b_mask])
    
    images.write(os.path.join(processed_data_path,"images.txt"))
    masks.write(os.path.join(processed_data_path,"masks.txt"))