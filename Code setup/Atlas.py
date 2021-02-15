import numpy as np 

def votingbased(masklist,rows=1024,columns=1024):
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
#a = np.array([[1,0,0],[1,0,0],[1,0,0]])
#b = np.array([[1,0,0],[1,0,0],[1,0,0]])
#c = np.array([[0,0,1],[0,0,1],[0,0,1]])

#listofmasks = [a,b,c]
#newmask = votingbased(listofmasks, rows = 3, columns = 3)
#print(newmask)
