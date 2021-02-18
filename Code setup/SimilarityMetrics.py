import numpy as np
import SimpleITK as sitk
import scipy.spatial
import medpy.metric
from config import data_path
import os

"""
Similarity metrics using medpy library
"""
def medpyDC(result, reference):
    dicecoefficient = medpy.metric.binary.dc(result, reference)

    return dicecoefficient

def medpyHD(result, reference):
    hausdorffdistance = medpy.metric.binary.hd(result, reference)

    return hausdorffdistance

def medpyRVD(result, reference):
    ravd = medpy.metric.binary.ravd(result, reference)
    ravd = ravd*100

    return ravd


"""
Manually programmed similarity metrics
To do: Add Hausdorff distance
"""

def getDSC(result, reference):    
    """Compute the Dice Similarity Coefficient."""
    #testArray   = sitk.GetArrayFromImage(testImage).flatten()
    #resultArray = sitk.GetArrayFromImage(resultImage).flatten()
    testImage = np.asarray(reference)
    resultImage = np.asarray(result)

    testImage = np.ndarray.flatten(testImage)
    resultImage = np.ndarray.flatten(resultImage)
    
    # similarity = 1.0 - dissimilarity
    return 1.0 - scipy.spatial.distance.dice(testImage, resultImage) 
    
def getDiceScore(result, reference, non_seg_score=1.0):
    """
    Computes the Dice coefficient between two masks
    Code largely copied from: https://gist.github.com/gergf/acd8e3fd23347cb9e6dc572f00c63d79    
    """
    true_mask = np.asarray(reference).astype(np.bool_)
    pred_mask = np.asarray(result).astype(np.bool_)

    # If both segmentations are all zero, the dice will be 1.
    im_sum = true_mask.sum() + pred_mask.sum()
    if im_sum == 0:
        return non_seg_score

    # Compute Dice coefficient
    intersection = np.logical_and(true_mask, pred_mask)
    y = 2. * intersection.sum() / im_sum
    return y

def getRVD(result, reference):   
    """Volume statistics."""

    vd = 100 * (result.sum() - reference.sum()) / float(reference.sum())

    return vd

"""
Used for mask operations
"""

def findfixedmask(fixednr, slicenr):
    fixed_mask_path = os.path.join(data_path,f"p{fixednr}\prostaat.mhd")
    fixed_mask = sitk.ReadImage(fixed_mask_path)
    fixed_mask_array_full = sitk.GetArrayFromImage(fixed_mask)
    fixed_mask_array = fixed_mask_array_full[slicenr,:,:]

    return fixed_mask_array
