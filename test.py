import elastix
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import SimpleITK as sitk
from scrollview import ScrollView



#image = sitk.ReadImage(r'C:\Users\s166646\Downloads\Capita Selecta\results2\result.0.mhd')
image = sitk.ReadImage(r'C:\Users\s166646\Downloads\TrainingData\TrainingData\p102\mr_bffe.mhd')
#image = sitk.ReadImage(r'C:\Users\s166646\Downloads\TrainingData\TrainingData\p102\prostaat.mhd')
image_array = sitk.GetArrayFromImage(image)

#print(image_array)
#print(np.max(image_array))

fig, ax = plt.subplots()
ScrollView(image_array).plot(ax)

#fixed_image = sitk.ReadImage(r'C:\Users\s166646\Downloads\ImagesforPractical\ImagesforPractical\chest_xrays\fixed_image.mhd')
#moving_image = sitk.ReadImage(r'C:\Users\s166646\Downloads\ImagesforPractical\ImagesforPractical\chest_xrays\moving_image.mhd')

#fixed_image_path = sitk.GetArrayFromImage(fixed_image)
#moving_image_path = sitk.GetArrayFromImage(moving_image)


#fig, ax = plt.subplots(1, 3, figsize=(20, 5))
#ax[0].imshow(fixed_image_path)
#ax[1].imshow(moving_image_path)
#ax[2].imshow(image_array[24], cmap='gray')

plt.show()