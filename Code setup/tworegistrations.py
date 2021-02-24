from transform import transform2d, transform3d
from visualize import visualize2d, visualize3d
from register import register2d, register3d
from changeparameters import changetxttuples
from SimilarityMetrics import medpyDC, medpyHD, findfixedmask, findtransformedmask
import os 
import matplotlib.pyplot as plt
import SimpleITK as sitk
import elastix
from config import data_path, elastix_path, transformix_path

from transform import transform2d, transform3d
from visualize import visualize2d, visualize3d
from register import register2d, register3d

movingnr = 120
fixednr = 107
slicenr = 30

parameter_file1 = 'Affine_parameters.txt'
parameter_file2 = 'Bspline_parameters.txt'

#register commands
#runnr = register2d(fixednr,movingnr,slicenr,parameter_file)
runnr = register3d(fixednr,movingnr,parameter_file1)

#Before this step: change offset in parameterfile from fixed to moving image!
#runnr = 20

#transform command
#transform2d(movingnr,slicenr,runnr,transformmask=True)
transform3d(movingnr,runnr,transformmask=True)

#visualisation commands
#visualize2d(fixednr,movingnr,slicenr,runnr)
#visualize3d(fixednr,movingnr,slicenr,runnr)

fixedmask = findfixedmask(107)
transformedmask = findtransformedmask(18)

print(medpyDC(transformedmask, fixedmask))
print(medpyHD(transformedmask, fixedmask))

fixed_image_path = os.path.join(f"results{runnr}","result.mhd")
moving_image_path = os.path.join(data_path,f"p{movingnr}\mr_bffe.mhd")

output_dir = f'results{runnr}'
    
el = elastix.ElastixInterface(elastix_path=elastix_path)

#run model
el.register(fixed_image=fixed_image_path,
    moving_image=moving_image_path,
    parameters=[parameter_file2],
    output_dir=output_dir)

transformedmask2 = findtransformedmask(18)

print(medpyDC(transformedmask, fixedmask))
print(medpyHD(transformedmask, fixedmask))
print(medpyDC(transformedmask2, fixedmask))
print(medpyHD(transformedmask2, fixedmask))