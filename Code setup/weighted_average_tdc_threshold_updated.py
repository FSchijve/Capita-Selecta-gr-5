#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 13:06:22 2021

@author: rooskraaijveld
"""

"""
ATLAS
"""
# CALCULATE THE WEIGHTED AVERAGE BETWEEN 3D IMAGES.
# THE TWO DICE COEFFICIENT THRESHOLDS:
# A. WHICH MASKS WE TAKE WITH US DURING CALCULATION
#threshold1 = 0.5
# B. THRESHOLD PER PIXEL, FOR FINAL MASK
#threshold2 = 0.5
#Change these two thresholds accordingly...

def votingbased_DCweighted_3d_all(masklist, DCscore, threshold1=0.5, threshold2=0.5):
    
    #put the z dimension at the begining => DC per patient or DC per slice of patient???
    #if the slices are made of zeroes, we should not count it either...
    slices = len(masklist[0])
    rows = len(masklist[0][0])
    columns = len(masklist[0][0][0])
    masklist_weighted = []
    new_masklist = []
    new_DCscore = []
    temp=[]
    
    
    # Step 1: keep masks with relevant dice score above threshold
    for k in range(len(DCscore)):
          if DCscore[k]>threshold1:
            new_masklist.append(masklist[k])
            new_DCscore.append(DCscore[k])  
    
    if len(new_masklist) == 0: raise Exception("No mask has dice score > threshold1 ("+str(threshold1)+")")

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
                    if newmask[z][x][y]>threshold2:
                        newmask[z][x][y]=1
                    else:
                        newmask[z][x][y]=0
    return newmask

#-------------- VOTING PER SLICE ----------------------------------------------

def votingbased_DCweighted_3d_slice(masklist, DCscore, threshold1=0.5, threshold2=0.5):
    patients = len(masklist) 
    slices = len(masklist[0])
    rows = len(masklist[0][0])
    columns = len(masklist[0][0][0])
    final_slice_mask = []
    
    
    for i in range(slices):
        masklist_weighted = []
        new_DCscore = []
        new_masklist = []

        for j in range(patients):
             # Step 1: Check if DC of slice of patient is above 0.5 
             if (DCscore[j][i] > threshold1):
                 new_masklist.append(masklist[j][i])
                 new_DCscore.append(DCscore[j][i])  
                 
        #print(sum(new_DCscore))
                 
    # Step 1.5: extra step if no image has value > threshold
        if len(new_masklist) == 0:
            maxDice = 0
            for j in range(patients):
                 new_masklist.append(masklist[j][i])
                 new_DCscore.append(DCscore[j][i])
                 if DCscore[j][i] > maxDice: maxDice = DCscore[j][i]
            print("Maximum weight of a slice was "+str(round(maxDice,3))+", lower then threshold "+str(threshold1)+". Threshold is ignored.")

    # Step 2: Do the weighting
    # I put this outside of the previous for-loop so that sum(new_DCscore will work)
        for j in range(len(new_masklist)):
            #print(new_DCscore[j])
            masklist_weighted.append((new_DCscore[j]*new_masklist[j])/sum(new_DCscore))
    
    # Step 3: Add the masks together
        summedmask = masklist_weighted[0]
        for n,item in enumerate(masklist_weighted):
            if n == 0: continue
            summedmask += item
        
        
    # Step 4: 
        newmask=summedmask
        for x in range(rows):
            for y in range(columns):
                     if newmask[x][y]>threshold2:
                         newmask[x][y]=1
                     else:
                         newmask[x][y]=0   
            
        final_slice_mask.append(newmask) # Stores the weighted mask from this iteration in a final variable 
    
    return final_slice_mask         
            
        
        
     

#-------------- EXAMPLE OF INPUT ----------------------------------------------
'''
a = np.array(
    [[[1, 0, 0], [1, 0, 0], [1, 0, 0]],
    [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
    [[1, 0, 0], [1, 0, 0], [1, 0, 0]]])


b = np.array(
    [[[0, 0, 1], [0, 0, 1], [0, 0, 1]],
    [[0, 0, 1], [0, 0, 1], [0, 0, 1]],
    [[0, 0, 1], [0, 0, 1], [0, 0, 1]]])


listofmasks = [a,b]
#newmask = votingbased2d(listofmasks, rows = 3, columns = 3)
#print(newmask)
#dicescore_example = [0.6,0.2,0.8]

DC_a = [0.6, 0.8, 0.6]
DC_b = [0.8, 0.6, 0.8]
DC_all = [DC_a, DC_b]
newmask_DC = votingbased_DCweighted_3d_slice(listofmasks,DC_all)
'''
#plt.imshow(newmask_DC[40,:, :], cmap='gray')