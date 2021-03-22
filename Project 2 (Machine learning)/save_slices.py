import torch
import numpy as np
import os
import torchvision
import random
import nibabel as nib
import gryds

#%%

# Normalization

def normalize_img(img): 
    # Enable when normalizing all values between 0 and 1:
    img=img/np.amax(img)
    
    # Enable when normalizing mean 0 std 1:
    #img = (img - np.mean(img))/np.std(img)
    
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
    img = img[:,:,0]
    mask = mask[:,:,0]

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
    img = img[:,:,0]
    mask = mask[:,:,0]

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
    img = img[:,:,0]
    mask = mask[:,:,0]

    img = torch.from_numpy(img.copy())
    mask = torch.from_numpy(mask.copy())
    flipped_img = torchvision.transforms.functional.hflip(img = img) # change to .vflip for vertical flip
    flipped_mask = torchvision.transforms.functional.hflip(img = mask)
    flipped_img = flipped_img.cpu().detach().numpy()
    flipped_mask = flipped_mask.cpu().detach().numpy()
    return flipped_img, flipped_mask

def Bspline_and_Affine(img, mask):
    img = img[:,:,0]
    mask = mask[:,:,0]

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
    img = img[:,:,0]
    mask = mask[:,:,0]

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

class Dataset():
    # Single dataset with slices
    def __init__(self, image_paths = []):
        self.image_paths = image_paths.copy()
        
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
        with open(filename,'w') as f:
            for image_path in self.image_paths:
                f.writelines(image_path+'\n')
        
    def read(self, filename):
        with open(filename,'r') as f:
            lines = f.readlines()

        for line in lines:
            self.image_paths.append(line[:-1])            
        
class XY_dataset():
    # Combined dataset (images + masks)
    def __init__(self, x_set, y_set, batch_size = 1, end_evaluation = False):
        if len(x_set) != len(y_set):
            raise Exception("Length of x_set is not the same as length of y_set.")
        self.x_set, self.y_set = x_set, y_set
        self.n = 0
        self.max = len(x_set)-1
        self.batch_size = batch_size
        self.end_evaluation = end_evaluation
        
    def __iter__(self):
        return self
    
    def __next__(self):
        
        if self.n + self.batch_size > self.max+1:
            self.n = 0
            if self.end_evaluation:
                raise StopIteration

        #print(f"\nRead {self.n} - {self.n+self.batch_size}")
        
        x_array, y_array = [], []
        for i in range(self.n,self.n+self.batch_size):
            x_array.append(self.x_set[i])
            y_array.append(self.y_set[i])
    
        self.n += self.batch_size

        return (np.array(x_array),np.array(y_array))

    def __len__(self):
        return len(self.x_set)


    #%%
if __name__ == '__main__':
    # Opening external dataset
    data_path = r"C:\Users\Dell\Documents\Medical_Imaging\CSMI_TUE\data\new"
    processed_data_path = r"C:\Users\Dell\Documents\Medical_Imaging\CSMI_TUE\preprocessed_data"
    
    if not os.path.isdir(processed_data_path):
        os.mkdir(processed_data_path)
    
    # number_list COMPLETE, this list should be used in the end for all the patients
    number_list = ['00', '01', '02', '04', '06', '07', 10, 13, 14, 16, 17, 18, 20, 21, 24, 25, 28, 29, 31, 32, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47]
    
    # number_list SHORT, this list should be used just to check the code for the first 3 patients
    number_list = number_list[:2]                
    #%%
    
    i = 0
    image_nr = 0
    List_images = Dataset()
    List_masks = Dataset()
    list_val_images = Dataset()
    list_val_masks = Dataset()
    
    # loop the patients
    #for number in number_list:
    for number in number_list:   
        mask_path = os.path.join(data_path,f"labelsTr\prostate_{number}.nii.gz"); 
        img_path  = os.path.join(data_path,f"imagesTr\prostate_{number}.nii.gz")
        
        List_img0 = Dataset()
        List_img1 = Dataset()
        List_img2 = Dataset()
        List_img3 = Dataset()
        List_img4 = Dataset()
        List_img5 = Dataset()
        
        List_mask0 = Dataset()
        List_mask1 = Dataset()
        List_mask2 = Dataset()
        List_mask3 = Dataset()
        List_mask4 = Dataset()
        List_mask5 = Dataset()
    
        nr_slices = nib.load(mask_path).get_fdata().shape[2]
    
        # Loop the slices 
        for slice in range(nr_slices):
            mask = np.rot90(nib.load(mask_path).get_fdata()[:,:,slice])
            img  = np.rot90(nib.load(img_path).get_fdata()[:,:,slice,0]) 
            print ('Patient',number,'slice',slice)
            img = normalize_img(img)
            mask = normalize_mask(mask)
            
            img = np.reshape(img, (256,256,1))
            mask = np.reshape(mask, (256,256,1))
    
            # Original image
            np.save(os.path.join(processed_data_path,f"slice_{str(image_nr)}.npy"),img)
            List_img0.addimage(os.path.join(processed_data_path,f"slice_{str(image_nr)}.npy"))
            image_nr += 1
            
            np.save(os.path.join(processed_data_path,f"slice_{str(image_nr)}.npy"),mask)
            List_mask0.addimage(os.path.join(processed_data_path,f"slice_{str(image_nr)}.npy"))
            image_nr += 1
    
            
            # Bspline and Affine and flipped transformation
            img_bspline_affine_flipped, mask_bspline_affine_flipped = Bspline_and_Affine_flipped(img, mask)
            img_bspline_affine_flipped = np.reshape(img_bspline_affine_flipped, (256,256,1))
            mask_bspline_affine_flipped = np.reshape(mask_bspline_affine_flipped, (256,256,1))
            
            np.save(os.path.join(processed_data_path,f"slice_{str(image_nr)}.npy"),img_bspline_affine_flipped)
            List_img1.addimage(os.path.join(processed_data_path,f"slice_{str(image_nr)}.npy"))
            image_nr += 1
    
            np.save(os.path.join(processed_data_path,f"slice_{str(image_nr)}.npy"),mask_bspline_affine_flipped)
            List_mask1.addimage(os.path.join(processed_data_path,f"slice_{str(image_nr)}.npy"))
            image_nr += 1
    
                    
            # Bspline and Affine and flipped transformation second time
            img_bspline_affine_flipped1, mask_bspline_affine_flipped1 = Bspline_and_Affine_flipped(img, mask)
            img_bspline_affine_flipped1 = np.reshape(img_bspline_affine_flipped1, (256,256,1))
            mask_bspline_affine_flipped1 = np.reshape(mask_bspline_affine_flipped1, (256,256,1))
            
            np.save(os.path.join(processed_data_path,f"slice_{str(image_nr)}.npy"),img_bspline_affine_flipped1)
            List_img2.addimage(os.path.join(processed_data_path,f"slice_{str(image_nr)}.npy"))
            image_nr += 1
    
            np.save(os.path.join(processed_data_path,f"slice_{str(image_nr)}.npy"),mask_bspline_affine_flipped1)
            List_mask2.addimage(os.path.join(processed_data_path,f"slice_{str(image_nr)}.npy"))
            image_nr += 1
            
    
            # Bspline and Affine transformation
            img_bspline_affine, mask_bspline_affine = Bspline_and_Affine(img, mask)
            img_bspline_affine = np.reshape(img_bspline_affine, (256,256,1))
            mask_bspline_affine = np.reshape(mask_bspline_affine, (256,256,1))
            
            np.save(os.path.join(processed_data_path,f"slice_{str(image_nr)}.npy"),img_bspline_affine)
            List_img3.addimage(os.path.join(processed_data_path,f"slice_{str(image_nr)}.npy"))
            image_nr += 1
    
            np.save(os.path.join(processed_data_path,f"slice_{str(image_nr)}.npy"),mask_bspline_affine)
            List_mask3.addimage(os.path.join(processed_data_path,f"slice_{str(image_nr)}.npy"))
            image_nr += 1
    
            
            # Bspline and Affine transformation second time
            img_bspline_affine1, mask_bspline_affine1 = Bspline_and_Affine(img, mask)
            img_bspline_affine1 = np.reshape(img_bspline_affine1, (256,256,1))
            mask_bspline_affine1 = np.reshape(mask_bspline_affine1, (256,256,1))
            
            np.save(os.path.join(processed_data_path,f"slice_{str(image_nr)}.npy"),img_bspline_affine1)
            List_img4.addimage(os.path.join(processed_data_path,f"slice_{str(image_nr)}.npy"))
            image_nr += 1
    
            np.save(os.path.join(processed_data_path,f"slice_{str(image_nr)}.npy"),mask_bspline_affine1)
            List_mask4.addimage(os.path.join(processed_data_path,f"slice_{str(image_nr)}.npy"))
            image_nr += 1
            
            
            # Bspline transformation
            img_bspline, mask_bspline = Bspline(img,mask)
            img_bspline = np.reshape(img_bspline, (256,256,1))
            mask_bspline = np.reshape(mask_bspline, (256,256,1))
            
            np.save(os.path.join(processed_data_path,f"slice_{str(image_nr)}.npy"),img_bspline)
            List_img5.addimage(os.path.join(processed_data_path,f"slice_{str(image_nr)}.npy"))
            image_nr += 1
    
            np.save(os.path.join(processed_data_path,f"slice_{str(image_nr)}.npy"),mask_bspline)
            List_mask5.addimage(os.path.join(processed_data_path,f"slice_{str(image_nr)}.npy"))
            image_nr += 1
            
        List_images.adddatasets([List_img1,List_img2,List_img3,List_img4,List_img5])
        List_masks.adddatasets([List_mask1,List_mask2,List_mask3,List_mask4,List_mask5])
        list_val_images.adddataset(List_img0)
        list_val_masks.adddataset(List_mask0)    
        i += 1
    
    List_images.write(os.path.join(processed_data_path,"List_images.txt"))
    List_masks.write(os.path.join(processed_data_path,"List_masks.txt"))
    list_val_images.write(os.path.join(processed_data_path,"list_val_images.txt"))
    list_val_masks.write(os.path.join(processed_data_path,"list_val_masks.txt"))