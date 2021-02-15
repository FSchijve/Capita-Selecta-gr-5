from transform import transform, transformmask
from visualize import visualize2d, visualize3d
from register import register2d, register3d

movingnr = 102
fixednr = 107
slicenr = 32
parameter_file = 'parameters_samplespace_MR_2D.txt'

#register commands
#runnr = register2d(fixednr,movingnr,slicenr,parameter_file)
#runnr register3d(fixednr,movingnr,parameter_file)

#Before this step: change offset in parameterfile from fixed to moving image!
runnr = 5

#transform command
transform(movingnr,runnr)
#transformmask(movingnr,runnr)

#visualisation commands
#visualize2d(fixednr,movingnr,slicenr,runnr)
visualize3d(fixednr,movingnr,slicenr,runnr)