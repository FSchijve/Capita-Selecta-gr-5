import os
import config
from single_slice import selectslice
from transform import transform
from visualize import visualize2d, visualize3d
from register import register2d, register3d

register3d(107,102,'parameters_samplespace_MR.txt')
#selectslice(patientfixed,slicenr,data_path)
#transform("107_0",3)
#visualize2d(107,107,0,3)
#visualize3d(107,107,0,1)