import os
import elastix
from config import data_path, transformix_path
from changeparameters import replace, readparameter

def transform2d(movingnr,slicenr,runnr,parameter_file,transformmask = True):
    movingnr = str(movingnr) + '_' + str(slicenr)
    
    transform_path = os.path.join(f'results{runnr}', r'TransformParameters.0.txt')
    moving_image_path = os.path.join(data_path,f"p{movingnr}\mr_bffe.mhd")
    output_path = f'results{runnr}'

    #make a new transformix object tr
    tr = elastix.TransformixInterface(parameters=transform_path,
                                  transformix_path=transformix_path)

    #transform a new image with the transformation parameters
    tr.transform_image(moving_image_path, output_dir=output_path)

    #get the Jacobian matrix
    tr.jacobian_matrix(output_dir=output_path)

    #get the Jacobian determinant
    tr.jacobian_determinant(output_dir=output_path)

    # Get the full deformation field
    tr.deformation_field(output_dir=output_path)

    if transformmask:
        moving_image_path = os.path.join(data_path,f"p{movingnr}\prostaat.mhd")
    
        #Change BSplineInterpolationOrder in parameterfile for mask transform
        interporder = readparameter(transform_path,"FinalBSplineInterpolationOrder")
        replace(transform_path,"FinalBSplineInterpolationOrder",0)

        #make a new transformix object tr
        tr = elastix.TransformixInterface(parameters=transform_path,
                                      transformix_path=transformix_path)

        output_path += r'\transformedmask'
        if os.path.exists(output_path) is False:
            os.mkdir(output_path)

        #transform a new image with the transformation parameter
        tr.transform_image(moving_image_path, output_dir=output_path)

        #Change back BSplineInterpolationOrder in parameterfile
        replace(transform_path,"FinalBSplineInterpolationOrder",interporder)

def transform3d(movingnr,runnr,transformmask = True):    
    transform_path = os.path.join(f'results{runnr}', r'TransformParameters.0.txt')
    moving_image_path = os.path.join(data_path,f"p{movingnr}\mr_bffe.mhd")
    output_path = f'results{runnr}'

    #make a new transformix object tr
    tr = elastix.TransformixInterface(parameters=transform_path,
                                  transformix_path=transformix_path)

    #transform a new image with the transformation parameters
    tr.transform_image(moving_image_path, output_dir=output_path)

    #get the Jacobian matrix
    tr.jacobian_matrix(output_dir=output_path)

    #get the Jacobian determinant
    tr.jacobian_determinant(output_dir=output_path)

    # Get the full deformation field
    tr.deformation_field(output_dir=output_path)

    if transformmask:
        print("Transforming the mask as well")
        moving_image_path = os.path.join(data_path,f"p{movingnr}\prostaat.mhd")

        #Change BSplineInterpolationOrder in parameterfile for mask transform
        interporder = readparameter(transform_path,"FinalBSplineInterpolationOrder")
        replace(transform_path,"FinalBSplineInterpolationOrder",0)
        
        #make a new transformix object tr
        tr = elastix.TransformixInterface(parameters=transform_path,
                                      transformix_path=transformix_path)

        output_path += r'\transformedmask'
        if os.path.exists(output_path) is False:
            os.mkdir(output_path)

        #transform a new image with the transformation parameters
        tr.transform_image(moving_image_path, output_dir=output_path)
        
        #Change back BSplineInterpolationOrder in parameterfile
        replace(transform_path,"FinalBSplineInterpolationOrder",interporder)