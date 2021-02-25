from transform import transform2d, transform3d
from visualize import visualize2d, visualize3d
from register import register2d, register3d
from combineatlasses import createmodel2d, createmodel3d, validatemodel, runmodel

movingnr = 120
fixednr = 125
slicenr = 30
parameter_file = 'BSpline_parameters.txt'

#single registration commands
#runnr = register2d(fixednr,movingnr,slicenr,parameter_file,verbose=False)
#runnr = register3d(fixednr,movingnr,parameter_file,verbose=False)

#runnr = 0

#single transformation commands
#transform2d(movingnr,slicenr,runnr,transformmask=True)
#transform3d(movingnr,runnr,transformmask=True)

#single visualisation commands
#visualize2d(fixednr,movingnr,slicenr,runnr)
#visualize3d(fixednr,movingnr,slicenr,runnr)

patients = [102, 107, 108, 109, 115, 116, 117, 119, 120, 125, 127, 128, 129, 133, 135]

#Divide patients into different sets
atlasset = patients[0:10]
optimizeset = patients[0:10]
validationset = patients[10:15]

#For testing
#atlasset = [102,107]
#optimizeset = [116,117]
#validationset = [127,128]

threshold1 = 0.0 #The performance scores of the atlas images/slices
threshold2 = 0.5 #The amount of cumulative (normalized) performance needed to activate a pixel

#model creation commands
#modelnr = createmodel3d(atlasset,optimizeset,parameter_file,threshold1,threshold2)
#modelnr = createmodel2d(atlasset,optimizeset,parameter_file,threshold1,threshold2)

modelnr = 5

#model validation command
validatemodel(modelnr,validationset)

unknownset = [127,128] #In the end this should be the images from Josien.
#model run command
#runmodel(modelnr,unknownset)