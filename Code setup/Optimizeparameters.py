from transform import transform2d, transform3d
from visualize import visualize2d, visualize3d
from register import register2d, register3d
from changeparameters import changetxttuples
from SimilarityMetrics import medpyDC, medpyHD, findfixedmask
import os 
import matplotlib.pyplot as plt
import SimpleITK as sitk

"""
Examplescript to optimize parameters
Dice score coefficient and Hausdorff distance is computed to measure the quality of registration for each parameter setting.

1. Install medpy package via command: "pip install medpy"
2. Download changeparameters.py, SimilarityMetrics.py and parameter file which in this case is 'Bspline_parameters.txt'
3. For each experiment, Change the 'listofparameters' and 'listoftuples' variables to the experiment settings you want, 
   in the comments I try to explain how these should be structured
"""
movingnr = 120
fixednr = 115
slicenr = 30

parameter_file = 'BSpline_parameters.txt'
fixed_mask_array = findfixedmask(fixednr, slicenr)

#In this experiment, the parameter '(FinalGridSpacingInPhysicalUnits 64.0)' is varied from value 64 to 2 by dividing the number by 2 every time
#These are the values that are going to be tested, update this to the numbers you're going to be testing as this will be used later 
listofparameters = ['64', '32', '16', '8', '4', '2']

#Varying the parameters happens down here. It's a list of tuples, where each tuple has three inputs: 1) The .txt file to be edited, 
# 2)The string that should be edited and 3) the string that should take the place of 2).
#In this case, the list is 6 tuples long, so 6 registrations are going to be performed. 

#Important! Because the script replaces all instances of the second input 2), these should be unique values that only appear once in the whole .txt file.
#These unique values can be created manually by editing these values in the parameter file first e.g. changing a value to 1.2 and then using
# ('Bspline_parameters.txt', '1.2', '2.2') to ensure that only this parameter is varied and not all 1's in the file.
listoftuples = [('BSpline_parameters.txt','64','64'), 
('BSpline_parameters.txt','64','32'), 
('BSpline_parameters.txt','32' ,'16'), 
('BSpline_parameters.txt','16' ,'8.001'), 
('BSpline_parameters.txt','8.001' ,'4.001'),
('BSpline_parameters.txt','4.001' ,'2.001')]
#Note the usage of values like 4.001 as to not accidentally change other parameter settings that have value 4
#Also make sure that the first value, in this case 64, is present and unique in the .txt file

listofDCs=[]
listofHDs=[]

for x in listoftuples:
    changetxttuples(x)
    runnr = register2d(fixednr,movingnr,slicenr,parameter_file)
    transform2d(movingnr,slicenr,runnr,transformmask=True)

    transformed_mask_path = os.path.join(f"results{runnr}",r"transformedmask\result.mhd")
    transformed_mask = sitk.ReadImage(transformed_mask_path)
    transformed_mask_array = sitk.GetArrayFromImage(transformed_mask)

    DCscore = medpyDC(transformed_mask_array, fixed_mask_array)
    HDscore = medpyHD(transformed_mask_array, fixed_mask_array)

    listofDCs.append(DCscore)
    listofHDs.append(HDscore)

print(listofparameters)
print(listofDCs)
print(listofHDs)

fig, ax = plt.subplots(1,2)
ax[0].plot(listofparameters, listofDCs)
ax[0].set_title('Dice coefficient scores')

ax[1].plot(listofparameters, listofHDs)
ax[1].set_title('Hausdorff distance scores')

plt.show()