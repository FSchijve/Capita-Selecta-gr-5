import SimpleITK as sitk
import os
from config import data_path

def selectslice(patientnr,slicenr):
    #Create a file with a single slice

    #import image
    image_path = os.path.join(data_path,f"p{patientnr}\mr_bffe.mhd")
    mask_path = os.path.join(data_path,f"p{patientnr}\prostaat.mhd")
    image = sitk.ReadImage(image_path)
    mask = sitk.ReadImage(mask_path)

    new_image_path = os.path.join(data_path,f"p{patientnr}_{slicenr}\mr_bffe.mhd")
    new_mask_path = os.path.join(data_path,f"p{patientnr}_{slicenr}\prostaat.mhd")

    if os.path.exists(os.path.join(data_path,f"p{patientnr}_{slicenr}")) is False:
        os.mkdir(os.path.join(data_path,f"p{patientnr}_{slicenr}"))

    writer = sitk.ImageFileWriter()
    writer.SetFileName(new_image_path)
    writer.Execute(image[:,:,slicenr])
    writer.SetFileName(new_mask_path)
    writer.Execute(mask[:,:,slicenr])
    
selectslice(102,0)