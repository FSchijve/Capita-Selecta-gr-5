import numpy as np 
#import Optimizeparameters
def votingbased2d(masklist,rows=1024,columns=1024):
    """
    Function that takes multiple masks as input, and returns a new mask based on the votes on each pixel of each individual mask
    The input should be a list of the masks, e.g. [mask1, mask2, mask3]
    The amount of rows and columns should be specified, standard they are set to 1024 and 1024, which is the shape of the images.
    To do: Make a choice of what to do when votes on a pixel are equal. Right now equal votes result in a 1
    """
    newmask = []
    for x in range(rows):
        for y in range(columns):
            zeros = []
            ones = []
            for z in masklist:
                if z[x,y]==1:
                    ones.append(1)
                elif z[x,y]==0:
                    zeros.append(0)
            if len(zeros)>len(ones):
                newmask.append(0)
            #We have to choose what to do if the votes are equal, right now equal votes result in a 1 
            elif len(zeros)==len(ones):
                newmask.append(1)
            elif len(ones)>len(zeros):
                newmask.append(1)

    newmask = np.reshape(newmask,(rows,columns))

    return newmask
        
# Example, uncomment to run 
a = np.array([[[1,0,0],[1,0,0],[1,0,0]]])
b = np.array([[[1,0,0],[1,0,0],[1,0,0]]])
c = np.array([[[0,0,1],[0,0,1],[0,0,1]]])

listofmasks = [a,b,c]
#newmask = votingbased2d(listofmasks, rows = 3, columns = 3)
#print(newmask)
dicescore_example = [0.6,0.2,0.8]

#def votingbased_DCweighted_2d(masklist, DCscore, rows=1024,columns=1024):
#    """
#    Function that takes multiple masks as input, and returns a new mask based on the votes on each pixel of each individual mask
#    The input should be a list of the masks, e.g. [mask1, mask2, mask3]
#    The amount of rows and columns should be specified, standard they are set to 1024 and 1024, which is the shape of the images.
#    To do: Make a choice of what to do when votes on a pixel are equal. Right now equal votes result in a 1
#    """
#    masklist_weighted = masklist
#    for i in range(len(DCscore)):
#        masklist_weighted[i]=DCscore[i]*masklist[i]/sum(DCscore)
#            
#    summedmask = masklist_weighted[0]
#    for j,item in enumerate(masklist_weighted):
#        if j == 0: continue
#        summedmask += item
#
#    newmask=summedmask
#    for x in range(rows):
#        for y in range(columns):
#            if newmask[x][y]>0.5:
#                newmask[x][y]=1
#            else:
#                newmask[x][y]=0
           
    #newmasklist = np.reshape(newmasklist,(len(newmasklist)rows,columns))

    #return newmask

#newmask_DC = votingbased_DCweighted_2d(listofmasks,dicescore_example)
#print(newmask_DC)

def votingbased_DCweighted_3d(masklist, DCscore, rows=1024,columns=1024):
    
    slices = len(masklist[0])
    rows = len(masklist[0][0])
    columns = len(masklist[0][0][0])
    masklist_weighted = masklist
    for k in range(len(DCscore)):
    #    if DCscore[k]>0.5:
    #        DCscore[k]=[]
    for i in range(len(DCscore)):
        masklist_weighted[i]=DCscore[i]*masklist[i]/sum(DCscore)
            
    summedmask = masklist_weighted[0]
    for j,item in enumerate(masklist_weighted):
        if j == 0: continue
        summedmask += item

    newmask=summedmask
    for x in range(rows):
        for y in range(columns):
            for z in range(slices):
                if newmask[z][x][y]>0.5:
                   newmask[z][x][y]=1
                else:
                   newmask[z][x][y]=0
           
    #newmasklist = np.reshape(newmasklist,(len(newmasklist)rows,columns))

    return newmask
    
newmask_DC = votingbased_DCweighted_3d(listofmasks,dicescore_example)
print(newmask_DC)