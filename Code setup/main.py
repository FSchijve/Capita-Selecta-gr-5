from transform import transform
from visualize import visualize2d, visualize3d
from register import register2d, register3d

#register2d(107,102,0,'parameters_samplespace_MR_2D.txt')

#Before this step: change offset in parameterfile from fixed to moving image!

transform(102,7)
visualize2d(107,102,0,7)
#visualize3d(107,102,50,6)