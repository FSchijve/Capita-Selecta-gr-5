# -*- coding: utf-8 -*-
"""
ATLAS

"""
import numpy as np 
import SimpleITK as sitk
import matplotlib.pyplot as plt
import elastix
import os
import imageio
import numpy as np

# CALCULATE THE WEIGHTED AVERAGE BETWEEN 3D IMAGES.
        # Default threshold value 0.5

def votingbased_DCweighted_3d_all(masklist, DCscore, rows=1024,columns=1024, threshold=0.5):
    
    #put the z dimension at the begining => DC per patient or DC per slice of patient???
    #if the slices are made of zeroes, we should not count it either...
    slices = len(masklist[0])
    rows = len(masklist[0][0])
    columns = len(masklist[0][0][0])
    masklist_weighted = []
    new_masklist = []
    new_DCscore = []
    temp=[]
    j=0
    
    # Step 1: keep masks with relevant dice score above threshold
    for k in range(len(DCscore)):
          if DCscore[k]>threshold:
            new_masklist.append(masklist[k])
            new_DCscore.append(DCscore[k])  
    
    if len(new_masklist) == 0: raise Exception("No mask has dice score > threshold ("+str(threshold)+")")

    # Step 2: do the weighting
    for i in range(len(new_DCscore)):
        temp.append(new_DCscore[i]*new_masklist[i])
        masklist_weighted.append(temp[i]/sum(new_DCscore))
                        
    # Step 3: add masks together         
    summedmask = masklist_weighted[0]
    for j,item in enumerate(masklist_weighted):
        if j == 0: continue
        summedmask += item
    
    # Step 4: keep relevant weightings
    newmask=summedmask
    for x in range(rows):
        for y in range(columns):
                for z in range(slices): 
                    if newmask[z][x][y]>0.5:
                        newmask[z][x][y]=1
                    else:
                        newmask[z][x][y]=0
    #print(newmask)       
    # newmasklist = np.reshape(newmasklist,(len(newmasklist)rows,columns))
    return newmask


#-------------- EXAMPLE OF INPUT ----------------------------------------------

# listofmasks = [mask_of_fixed_array, mask_of_moving_array]

# dicescore_example = [0.8, 0.51]

# newmask_DC = votingbased_DCweighted_3d_all(listofmasks,dicescore_example)

# plt.imshow(newmask_DC[40,:, :], cmap='gray')

