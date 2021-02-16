from transform import transform2d, transform3d
from visualize import visualize2d, visualize3d
from register import register2d, register3d

movingnr = 120
fixednr = 125
slicenr = 30
parameter_file = 'parameters_samplespace_MR_2D.txt'

#register commands
#runnr = register2d(fixednr,movingnr,slicenr,parameter_file)
#runnr register3d(fixednr,movingnr,parameter_file)

#Before this step: change offset in parameterfile from fixed to moving image!
runnr = 5

#transform command
#transform2d(movingnr,slicenr,runnr,transformmask=True)
#transform3d(movingnr,runnr,transformmask=True)

#visualisation commands
#visualize2d(fixednr,movingnr,slicenr,runnr)
visualize3d(fixednr,movingnr,slicenr,runnr)