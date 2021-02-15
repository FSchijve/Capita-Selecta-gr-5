from transform import transform
from visualize import visualize2d, visualize3d
from register import register2d, register3d

movingnr = 102
fixednr = 107
slicenr = 0
parameter_file = 'parameters_samplespace_MR_2D.txt'

#register commands
#runnr = register2d(fixednr,movingnr,slicenr,parameter_file)
#runnr register3d(fixednr,movingnr,parameter_file)

#Before this step: change offset in parameterfile from fixed to moving image!
runnr = 7

#transform command
transform(movingnr,runnr)

#visualisation commands
visualize2d(fixednr,movingnr,slicenr,runnr)
#visualize3d(fixednr,movingnr,slicenr,runnr)