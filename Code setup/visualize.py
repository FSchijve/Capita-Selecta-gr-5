import matplotlib.pyplot as plt
import os
import imageio
from config import data_path

def visualize2d(fixednr,movingnr,slicenr,runnr):
    fixednr = str(fixednr) + '_' + str(slicenr)
    movingnr = str(movingnr) + '_' + str(slicenr)

    fixed_image_path = os.path.join(data_path,f"p{fixednr}\mr_bffe.mhd")
    moving_image_path = os.path.join(data_path,f"p{movingnr}\mr_bffe.mhd")
    result_path = os.path.join(f'results{runnr}', r'result.mhd')
    jacobian_determinant_path = os.path.join(f'results{runnr}', r'fullSpatialJacobian.mhd')
    def_field_path = os.path.join(f'results{runnr}', r'deformationField.mhd')

    fig, ax = plt.subplots(1, 5, figsize=(15, 5))
    ax[0].imshow(imageio.imread(fixed_image_path), cmap='gray')
    ax[0].set_title('Fixed image')
    ax[1].imshow(imageio.imread(moving_image_path), cmap='gray')
    ax[1].set_title('Moving image')
    ax[2].imshow(imageio.imread(result_path), cmap='gray')
    ax[2].set_title('Transformed\nmoving image')
    ax[3].imshow(imageio.imread(jacobian_determinant_path)[:,:,0])
    ax[3].set_title('Jacobian\ndeterminant')
    ax[4].imshow(imageio.imread(def_field_path)[:,:,0])
    ax[4].set_title('Deformation\nfield')
    
def visualize3d(fixednr,movingnr,slicenr,runnr):
    fixed_image_path = os.path.join(data_path,f"p{fixednr}\mr_bffe.mhd")
    moving_image_path = os.path.join(data_path,f"p{movingnr}\mr_bffe.mhd")
    result_path = os.path.join(f'results{runnr}', r'result.mhd')
    jacobian_determinant_path = os.path.join(f'results{runnr}', r'fullSpatialJacobian.mhd')
    def_field_path = os.path.join(f'results{runnr}', r'deformationField.mhd')

    fig, ax = plt.subplots(1, 5, figsize=(15, 5))
    ax[0].imshow(imageio.imread(fixed_image_path)[slicenr,:,:], cmap='gray')
    ax[0].set_title('Fixed image')
    ax[1].imshow(imageio.imread(moving_image_path)[slicenr,:,:], cmap='gray')
    ax[1].set_title('Moving image')
    ax[2].imshow(imageio.imread(result_path)[slicenr,:,:], cmap='gray')
    ax[2].set_title('Transformed\nmoving image')
    ax[3].imshow(imageio.imread(jacobian_determinant_path)[slicenr,:,:,0])
    ax[3].set_title('Jacobian\ndeterminant')
    ax[4].imshow(imageio.imread(def_field_path)[slicenr,:,:,0])
    ax[4].set_title('Deformation\nfield')