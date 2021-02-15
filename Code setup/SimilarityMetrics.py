import numpy as np
    
def getDiceScore(true_mask, pred_mask, non_seg_score=1.0):
    """
    Computes the Dice coefficient between two masks
    Code largely copied from: https://gist.github.com/gergf/acd8e3fd23347cb9e6dc572f00c63d79    
    """
    true_mask = np.asarray(true_mask).astype(np.bool_)
    pred_mask = np.asarray(pred_mask).astype(np.bool_)

    # If both segmentations are all zero, the dice will be 1.
    im_sum = true_mask.sum() + pred_mask.sum()
    if im_sum == 0:
        return non_seg_score

    # Compute Dice coefficient
    intersection = np.logical_and(true_mask, pred_mask)
    y = 2. * intersection.sum() / im_sum
    return y

#Example lines, uncomment to test
#A = [[0,1,1,1,1,0],[0,1,0,0,1,0],[0,1,0,0,1,0],[0,1,0,0,1,0],[0,1,1,1,1,0]]
#B = [[0,1,1,1,1,0],[0,1,0,0,1,0],[0,1,0,0,1,0],[0,1,0,0,1,0],[0,1,1,1,1,0]]
#C = [[1,1,1,1,0,0],[1,0,0,0,1,0],[1,0,0,0,1,0],[1,0,0,0,1,0],[1,1,1,1,0,0]]

#scoreAB = getDiceScore(A,B)
#scoreAC = getDiceScore(A,C)
#print(scoreAB)
#print(scoreAC)
