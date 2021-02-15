import os
import elastix
from config import data_path, transformix_path

def transform(movingnr,runnr):
    transform_path = os.path.join(f'results{runnr}', r'TransformParameters.0.txt')
    moving_image_path = os.path.join(data_path,f"p{movingnr}\mr_bffe.mhd")

    #make a new transformix object tr
    tr = elastix.TransformixInterface(parameters=transform_path,
                                  transformix_path=transformix_path)

    #transform a new image with the transformation parameters
    tr.transform_image(moving_image_path, output_dir=f'results{runnr}')

    #get the Jacobian matrix
    tr.jacobian_matrix(output_dir=f'results{runnr}')

    #get the Jacobian determinant
    tr.jacobian_determinant(output_dir=f'results{runnr}')

    # Get the full deformation field
    tr.deformation_field(output_dir=f'results{runnr}')

def transformmask(movingnr,runnr):
    transform_path = os.path.join(f'results{runnr}', r'TransformParameters.0.txt')
    moving_image_path = os.path.join(data_path,f"p{movingnr}\prostaat.mhd")

    #make a new transformix object tr
    tr = elastix.TransformixInterface(parameters=transform_path,
                                  transformix_path=transformix_path)

    output_dir = f'results{runnr}'
    output_dir += r'\transformedmask'
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)

    #transform a new image with the transformation parameters
    tr.transform_image(moving_image_path, output_dir=output_dir)