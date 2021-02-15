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
