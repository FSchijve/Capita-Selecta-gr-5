from transform import transform2d, transform3d
from visualize import visualize2d, visualize3d
from register import register2d, register3d
from changeparameters import replace, readparameter
from SimilarityMetrics import medpyDC, medpyHD, findfixedmask2d
import os 
import matplotlib.pyplot as plt
import SimpleITK as sitk

movingnr = 120
fixednr = 115
slicenr = 30

parameter_file = 'BSpline_parameters.txt'
keyword = 'FinalGridSpacingInPhysicalUnits'
listofvalues = ['64', '32', '16', '8', '4', '2']

listofDCs=[]
listofHDs=[]
fixed_mask_array = findfixedmask2d(fixednr, slicenr)

for x in listofvalues:
    replace(parameter_file, keyword, x)
    runnr = register2d(fixednr,movingnr,slicenr,parameter_file)
    transform2d(movingnr,slicenr,runnr,transformmask=True)

    transformed_mask_path = os.path.join(f"results{runnr}",r"transformedmask\result.mhd")
    transformed_mask = sitk.ReadImage(transformed_mask_path)
    transformed_mask_array = sitk.GetArrayFromImage(transformed_mask)

    DCscore = medpyDC(transformed_mask_array, fixed_mask_array)
    HDscore = medpyHD(transformed_mask_array, fixed_mask_array)

    listofDCs.append(DCscore)
    listofHDs.append(HDscore)
    
print(keyword)
print(listofvalues)
print('Dice scores')
print(listofDCs)
print('Hausendorff distances')
print(listofHDs)

fig, ax = plt.subplots(1,2)
ax[0].plot(listofvalues, listofDCs)
ax[0].set_title('Dice coefficient scores')

ax[1].plot(listofvalues, listofHDs)
ax[1].set_title('Hausdorff distance scores')

plt.show()
