import elastix
import os
from pathlib import Path
from config import data_path, elastix_path, transformix_path
from single_slice import selectslice

def register3d(fixednr,movingnr,parameter_path):
    runnr = findnewrunnr()
    
    print("Run",runnr)
    print("------------------")

    #import images
    fixed_image_path = os.path.join(data_path,f"p{fixednr}\mr_bffe.mhd")
    moving_image_path = os.path.join(data_path,f"p{movingnr}\mr_bffe.mhd")

    #create model
    if not os.path.exists(elastix_path):
        raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
    if not os.path.exists(transformix_path):
        raise IOError('Transformix cannot be found, please set the correct TRANSFORMIX_PATH.')
    
    if os.path.exists(f'results{runnr}') is False:
        os.mkdir(f'results{runnr}')
    
    el = elastix.ElastixInterface(elastix_path=elastix_path)

    #run model
    el.register(fixed_image=fixed_image_path,
        moving_image=moving_image_path,
        parameters=[parameter_path],
        output_dir=f'results{runnr}')
    
def register2d(fixednr,movingnr,slicenr,parameter_path):
    selectslice(fixednr,slicenr)
    selectslice(movingnr,slicenr)

    fixednr = str(fixednr) + '_' + str(slicenr)
    movingnr = str(movingnr) + '_' + str(slicenr)
    runnr = findnewrunnr()
       
    print("Run",runnr)
    print("------------------")

    #import images
    fixed_image_path = os.path.join(data_path,f"p{fixednr}\mr_bffe.mhd")
    moving_image_path = os.path.join(data_path,f"p{movingnr}\mr_bffe.mhd")

    #create model
    if not os.path.exists(elastix_path):
        raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
    if not os.path.exists(transformix_path):
        raise IOError('Transformix cannot be found, please set the correct TRANSFORMIX_PATH.')
    
    if os.path.exists(f'results{runnr}') is False:
        os.mkdir(f'results{runnr}')
    
    el = elastix.ElastixInterface(elastix_path=elastix_path)

    #run model
    el.register(fixed_image=fixed_image_path,
        moving_image=moving_image_path,
        parameters=[parameter_path],
        output_dir=f'results{runnr}')
    
def findnewrunnr():
    #Find the runnumber of the new file

    #Find all current files
    p = Path('./')
    list_of_files = list(p.glob('**'))
    
    newrunnr = 0
    for i, item in enumerate(list_of_files): #Loop over files
        if str(item)[:7] != 'results': continue #If file is result file
        runnr = int(str(item)[7:]) #Find runnumber
        if runnr >= newrunnr: newrunnr = runnr + 1 #Define new runnnumber

    return newrunnr
