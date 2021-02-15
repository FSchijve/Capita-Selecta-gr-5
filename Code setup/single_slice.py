import SimpleITK as sitk
import os
from config import data_path

def selectslice(patientnr,slicenr):
    #Create a file with a single slice

    #import image
    image_path = os.path.join(data_path,f"p{patientnr}\mr_bffe.mhd")
    image = sitk.ReadImage(image_path)

    new_image_path = os.path.join(data_path,f"p{patientnr}_{slicenr}\mr_bffe.mhd")

    if os.path.exists(os.path.join(data_path,f"p{patientnr}_{slicenr}")) is False:
        os.mkdir(os.path.join(data_path,f"p{patientnr}_{slicenr}"))

    writer = sitk.ImageFileWriter()
    writer.SetFileName(new_image_path)
    writer.Execute(image[:,:,slicenr])