# -*- coding: utf-8 -*-


import SimpleITK as sitk
import matplotlib.pyplot as plt
import elastix
import os
import imageio

'''
Variables
''' 
#patient numbers
patientfixed = 102
patientmoving = 107

#data base path
data_path = r"C:\Users\giuli\Desktop\Uni Utrecht\Capita Selecta\Project\TrainingData"

ELASTIX_PATH     = os.path.join(r"C:\Users\giuli\Elastix\elastix.exe")
TRANSFORMIX_PATH = os.path.join(r"C:\Users\giuli\Elastix\transformix.exe")

runnr = 1


'''
Code
'''
#import images
# Paths of the fixed and moving images and the masks
fixed_image_path = os.path.join(data_path,f"p{patientfixed}\mr_bffe.mhd")
moving_image_path = os.path.join(data_path,f"p{patientmoving}\mr_bffe.mhd")

fixed_mask_path  = os.path.join(data_path,f"p{patientfixed}\prostaat.mhd")
moving_mask_path = os.path.join(data_path,f"p{patientmoving}\prostaat.mhd")

# Arrays of fixed and moving and masks
fixed_image_array  = sitk.GetArrayFromImage(sitk.ReadImage(fixed_image_path))
moving_image_array = sitk.GetArrayFromImage(sitk.ReadImage(moving_image_path))

mask_of_fixed_array  = sitk.GetArrayFromImage(sitk.ReadImage(fixed_mask_path))
mask_of_moving_array = sitk.GetArrayFromImage(sitk.ReadImage(moving_mask_path))

# Applying the masks
masked_fixed_array  = fixed_image_array  * mask_of_fixed_array;
masked_moving_array = moving_image_array * mask_of_moving_array;

# # Saving the images after applying the masks
masked_fixed = sitk.GetImageFromArray(masked_fixed_array)
masked_fixed_path= os.path.join(data_path, f"p{patientfixed}\masked_fixed.mhd")
sitk.WriteImage(masked_fixed, masked_fixed_path)

masked_moving = sitk.GetImageFromArray(masked_moving_array)
masked_moving_path= os.path.join(data_path, f"p{patientfixed}\masked_moving.mhd")
sitk.WriteImage(masked_moving, masked_moving_path)

## --------------- SCROLL SLICES -------------------------------------------#
class IndexTracker(object):
    
    def __init__(self, ax, masked_fixed_array):
          #def __init__(self, ax, img_name, . . . ) - if you want more images, copy paste the code again 
        
        self.ax0 = ax[0]
        ax[0].set_title('prostaat masked')
        
        self.masked_fixed_array = masked_fixed_array
        rows0, cols0, self.slices0 = masked_fixed_array.shape
        self.ind0 = self.slices0//2
    
        self.im0 = ax[0].imshow(self.masked_fixed_array[:, :, self.ind0], cmap='gray')
    
        self.update()
        
    
    def onscroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind0 = (self.ind0 + 1) % self.slices0
    
        else:
            self.ind0 = (self.ind0 - 1) % self.slices0
    
        self.update()
        
    def update(self):
        self.im0.set_data(self.masked_fixed_array[:, :, self.ind0])
        ax[0].set_ylabel('slice %s' % self.ind0)
        self.im0.axes.figure.canvas.draw()
        
    
# fig, ax = plt.subplots(1, 2)
# tracker = IndexTracker(ax, masked_fixed_array) #keep arrayyyyyyy
# fig.canvas.mpl_connect('scroll_event', tracker.onscroll)


#%%

# #create model
if not os.path.exists(ELASTIX_PATH):
    raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
if not os.path.exists(TRANSFORMIX_PATH):
    raise IOError('Transformix cannot be found, please set the correct TRANSFORMIX_PATH.')
    
    
el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)

parameter_path= os.path.join(data_path, 'parameters_bspline.txt')
result_folder_path= os.path.join(data_path, rf"p{patientfixed}\result")

if os.path.exists(result_folder_path) is False:
    os.mkdir(result_folder_path)


#run model
el.register(fixed_image=masked_fixed_path,
    moving_image=masked_moving_path,
    parameters=[parameter_path],
    output_dir=result_folder_path)


# open the logfile into the dictionary log
transform_path = os.path.join(result_folder_path, r'TransformParameters.0.txt')
result_path    = os.path.join(result_folder_path, r'result.0.mhd')



# #load result images
transformed_moving_image = sitk.ReadImage(result_path)
transformed_moving_image = sitk.GetArrayFromImage(transformed_moving_image)


fig, ax = plt.subplots(1, 2)
tracker = IndexTracker(ax, masked_fixed_array) #keep arrayyyyyyy
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)

#________subplot??? doesn't work yet________________
#fig, ax1 = plt.subplots(1, 2)
#tracker1 = IndexTracker(ax, masked_moving_array) #keep arrayyyyyyy
#fig.canvas.mpl_connect('scroll_event', tracker1.onscroll)

# Iteration_file_path = os.path.join(result_folder_path, r'IterationInfo.0.R0.txt')
# log = elastix.logfile(Iteration_file_path)

##plot log
# plt.plot(log['itnr'], log['metric'])

# tracker = IndexTracker(ax, masked_moving_array) #keep arrayyyyyyy
# fig.canvas.mpl_connect('scroll_event', tracker.onscroll)

# tracker3 = IndexTracker(ax, transformed_moving_image) #keep arrayyyyyyy
# fig.canvas.mpl_connect('scroll_event', tracker3.onscroll)

# #Sow resulting image side by side with fixed and moving image
# fig, ax = plt.subplots(1, 4, figsize=(15, 5))
# ax[0].imshow(fixed_image[viewslice,:,:], cmap='gray')
# ax[0].set_title('Fixed image')
# ax[1].imshow(moving_image[viewslice,:,:], cmap='gray')
# ax[1].set_title('Moving image')
# ax[2].imshow(transformed_moving_image[viewslice,:,:], cmap='gray')
# ax[2].set_title('Transformed\nmoving image')


# #make a new transformix object tr
# tr = elastix.TransformixInterface(parameters=transform_path,
#                                   transformix_path=TRANSFORMIX_PATH)

# #transform a new image with the transformation parameters
# transformed_image_path = tr.transform_image(moving_image_path, output_dir=f'results{runnr}')


#%%
# #get the Jacobian matrix
# jacobian_matrix_path = tr.jacobian_matrix(output_dir=f'results{runnr}')

# #get the Jacobian determinant
# jacobian_determinant_path = tr.jacobian_determinant(output_dir=f'results{runnr}')

# # Get the full deformation field
# deformation_field_path = tr.deformation_field(output_dir=f'results{runnr}')

# # Add a plot of the Jacobian determinant (in this case, the file is a tiff file)
# ax[3].imshow(imageio.imread(jacobian_determinant_path.replace('dcm', 'tiff'))[viewslice,:,:])
# ax[3].set_title('Jacobian\ndeterminant')

# # Show the plots
# [x.set_axis_off() for x in ax]
# plt.show()
