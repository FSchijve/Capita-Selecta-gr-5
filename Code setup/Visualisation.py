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
data_path = r"C:\Users\Dell\Documents\Medical_Imaging\CSMI_TUE\data"

ELASTIX_PATH = os.path.join(r'C:\Users\Dell\Documents\Medical_Imaging\CSMI_TUE\elastix.exe')
TRANSFORMIX_PATH = os.path.join(r'C:\Users\Dell\Documents\Medical_Imaging\CSMI_TUE\transformix.exe')

runnr = 0
viewslice = 31

'''
Code
'''
#import images
fixed_image_path = os.path.join(data_path,f"p{patientfixed}\mr_bffe.mhd")
moving_image_path = os.path.join(data_path,f"p{patientmoving}\mr_bffe.mhd")

fixed_image = sitk.GetArrayFromImage(sitk.ReadImage(fixed_image_path))
moving_image = sitk.GetArrayFromImage(sitk.ReadImage(moving_image_path))

transform_path = os.path.join(f'results{runnr}', r'TransformParameters.0.txt')
result_path = os.path.join(f'results{runnr}', r'result.0.mhd')



#load result images
transformed_moving_image = sitk.ReadImage(result_path)
transformed_moving_image = sitk.GetArrayFromImage(transformed_moving_image)


#Sow resulting image side by side with fixed and moving image
fig, ax = plt.subplots(1, 4, figsize=(15, 5))
ax[0].imshow(fixed_image[viewslice,:,:], cmap='gray')
ax[0].set_title('Fixed image')
ax[1].imshow(moving_image[viewslice,:,:], cmap='gray')
ax[1].set_title('Moving image')
ax[2].imshow(transformed_moving_image[viewslice,:,:], cmap='gray')
ax[2].set_title('Transformed\nmoving image')


#make a new transformix object tr
tr = elastix.TransformixInterface(parameters=transform_path,
                                  transformix_path=TRANSFORMIX_PATH)

#transform a new image with the transformation parameters
transformed_image_path = tr.transform_image(moving_image_path, output_dir=f'results{runnr}')

#get the Jacobian matrix
jacobian_matrix_path = tr.jacobian_matrix(output_dir=f'results{runnr}')

#get the Jacobian determinant
jacobian_determinant_path = tr.jacobian_determinant(output_dir=f'results{runnr}')

# Get the full deformation field
deformation_field_path = tr.deformation_field(output_dir=f'results{runnr}')

# Add a plot of the Jacobian determinant (in this case, the file is a tiff file)
ax[3].imshow(imageio.imread(jacobian_determinant_path.replace('dcm', 'tiff'))[viewslice,:,:])
ax[3].set_title('Jacobian\ndeterminant')

# Show the plots
[x.set_axis_off() for x in ax]
plt.show()