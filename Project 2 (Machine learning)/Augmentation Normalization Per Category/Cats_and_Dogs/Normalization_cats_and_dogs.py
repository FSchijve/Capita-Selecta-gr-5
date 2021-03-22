import os
import imageio
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

input_dir = "images/"
target_dir = "annotations/trimaps/"


#%% Normalize
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
    max_value = np.amax(mask)
    min_value = np.amin(mask)
    for i in range(128):
        for j in range(128):
            if mask[i,j]==max_value:
                mask[i,j]=1
            if mask[i,j]==min_value:
                mask[i,j]=1
            else:
                mask[i,j]=0

    return mask

#%% Making list of paths for all images and masks
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

#%% Some images gave errors (idk why), so I removed them manually
input_img_paths.remove('images/Abyssinian_34.jpg')
input_img_paths.remove('images/Egyptian_Mau_139.jpg')
input_img_paths.remove('images/Egyptian_Mau_145.jpg')
input_img_paths.remove('images/Egyptian_Mau_167.jpg')
input_img_paths.remove('images/Egyptian_Mau_177.jpg')
input_img_paths.remove('images/Egyptian_Mau_191.jpg')
target_img_paths.remove('annotations/trimaps/Abyssinian_34.png')
target_img_paths.remove('annotations/trimaps/Egyptian_Mau_139.png')
target_img_paths.remove('annotations/trimaps/Egyptian_Mau_145.png')
target_img_paths.remove('annotations/trimaps/Egyptian_Mau_167.png')
target_img_paths.remove('annotations/trimaps/Egyptian_Mau_177.png')
target_img_paths.remove('annotations/trimaps/Egyptian_Mau_191.png')

#%% Put all images and masks in a list
List_images = []
List_masks = []
runner = 0
for i in range(len(input_img_paths)):
    runner = runner+1
    print(runner)
    img = imageio.imread(input_img_paths[i])
    mask = imageio.imread(target_img_paths[i])
    
    # some images are different shapes (:,:,3) and some only (:,:)
    if len(img.shape)>2:
        img = img[:,:,0]
    
    # removing images that are very big, since resizing them to 128 x 128 will maybe get very bad
    if img.shape[0]>128:
        if img.shape[0]<512:
            if img.shape[1]>128:
                if img.shape[1]<512:
                    img = normalize_img(img)
                    mask = normalize_mask(mask)
                    List_images.append(img)
                    List_masks.append(mask)

#%% Plot one random animal + mask
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(List_images[10],cmap='gray')
ax[0].set_title('Image')
ax[1].imshow(List_masks[10],cmap='gray')
ax[1].set_title('Mask')
