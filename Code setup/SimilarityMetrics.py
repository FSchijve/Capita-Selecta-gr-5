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

def findtransformedmask(runnr):
    transformed_mask_path = os.path.join(f"results{runnr}",r"transformedmask\result.mhd")
    transformed_mask = sitk.ReadImage(transformed_mask_path)
    transformed_mask_array = sitk.GetArrayFromImage(transformed_mask)

    return transformed_mask_array

"""
Point clicker code
"""

def cpselect(imagePath1, imagePath2):
	# Pops up a matplotlib window in which to select control points on the two images given as input.
	#
	# Input:
    # imagePath1 - fixed image path
    # imagePath2 - moving image path
    # Output:
    # X - control points in the fixed image
    # Xm - control points in the moving image
	
	#load the images
	image1 = plt.imread(imagePath1)
	image2 = plt.imread(imagePath2)
	
	#ensure that the plot opens in its own window
	get_ipython().run_line_magic('matplotlib', 'qt')
	
	#set up the overarching window
	fig, axes = plt.subplots(1,2)
	fig.figsize = [16,9]
	fig.suptitle("Left Mouse Button to create a point.\n Right Mouse Button/Delete/Backspace to remove the newest point.\n Middle Mouse Button/Enter to finish placing points.\n First select a point in Image 1 and then its corresponding point in Image 2.")
	
	#plot the images
	axes[0].imshow(image1)
	axes[0].set_title("Image 1")
	
	axes[1].imshow(image2)
	axes[1].set_title("Image 2")
	
	#accumulate points
	points = plt.ginput(n=-1, timeout=30)
	plt.close(fig)
	
	#restore to inline figure placement
	get_ipython().run_line_magic('matplotlib', 'inline')
	
	#if there is an uneven amount of points, raise an exception
	if not (len(points)%2 == 0):
		raise Exception("Uneven amount of control points: {0}. Even amount of control points required.".format(len(points)))
		
	#if there are no points, raise an exception
	if not (len(points)> 0):
		raise Exception("No control points selected.")
	
	#subdivide the points into two different arrays. If the current number is even belongs to the first first image, and uneven to the second image. (Assuming the points were entered in the correct order.)
	#X and Y values are on rows, with each column being a pair of values.
	k = len(points)//2
	X = np.empty((2,k))
	X[:] = np.nan
	Xm = np.empty((2,k))
	Xm[:] = np.nan

	for i in np.arange(len(points)):
		if i%2 == 0 :
			X[0,i//2] = points[i][0]
			X[1,i//2] = points[i][1]
		else:
			Xm[0,i//2] = points[i][0]
			Xm[1,i//2] = points[i][1]
	
	return X, Xm
