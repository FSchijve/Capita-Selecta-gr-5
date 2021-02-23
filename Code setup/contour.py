import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
from config import data_path
import imageio


runnr = 79

movingnr = 120
fixednr = 125
slicenr = 65

moving_image_path = os.path.join(data_path,f"p{movingnr}\mr_bffe.mhd")
mask_reference_path = os.path.join(data_path, f"p{fixednr}\prostaat.mhd")
mask_calculated_path = os.path.join(f'results{runnr}',f'transformedmask','result.mhd')

moving_image = imageio.imread(moving_image_path)[slicenr,:,:]
mask_ref = imageio.imread(mask_reference_path)[slicenr,:,:]
mask_calc = imageio.imread(mask_calculated_path)[slicenr,:,:]
mask_calc = np.uint8(mask_calc)

## contour for reference mask
contours, hierarchy = cv.findContours(mask_ref.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE, offset=(0, 0))
mask_ref = cv.cvtColor(mask_ref, cv.COLOR_GRAY2BGR)
cv.drawContours(mask_ref, contours, -1,(255,0,0), 1)

## contour for calculated mask
contours1, hierarchy1 = cv.findContours(mask_calc.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE, offset=(0, 0))
mask_calc = cv.cvtColor(mask_calc, cv.COLOR_GRAY2BGR)
cv.drawContours(mask_calc, contours1, -1,(0,255,0), 1)

## plot of two contours plus real image
plt.figure(figsize = (10,10))
plt.imshow(moving_image)
plt.imshow(mask_ref, alpha = 0.35)
plt.imshow(mask_calc, alpha=0.2)


