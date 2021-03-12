import elastix
import os
from pathlib import Path
from config import data_path, elastix_path, transformix_path
from single_slice import selectslice

def register3d(fixednr,movingnr,parameter_path,runnr=-1,verbose=True):
    if runnr < 0:
        runnr = findnewrunnr()
    
    runprint = "Registration run " + str(runnr)
    if runnr > 99998: runprint = "Registration"
    if not verbose: runprint += ", moving = " + str(movingnr) + ", fixed = "+str(fixednr)
    print(runprint)
    if verbose: print("------------------")

    #import images
    fixed_image_path = os.path.join(data_path,f"p{fixednr}\mr_bffe.mhd")
    moving_image_path = os.path.join(data_path,f"p{movingnr}\mr_bffe.mhd")

    #create model
    if not os.path.exists(elastix_path):
        raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
    if not os.path.exists(transformix_path):
        raise IOError('Transformix cannot be found, please set the correct TRANSFORMIX_PATH.')
    
    output_dir = f'results{runnr}'
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)
    
    el = elastix.ElastixInterface(elastix_path=elastix_path)

    #run model
    el.register(fixed_image=fixed_image_path,
        moving_image=moving_image_path,
        parameters=[parameter_path],
        output_dir=output_dir,
        verbose=verbose)
    
    #writeoffset(moving_image_path, output_dir + r'\TransformParameters.0.txt')

    return runnr
    
def register2d(fixednr,movingnr,slicenr,parameter_path,runnr=-1,verbose=True):
    if runnr < 0:
        runnr = findnewrunnr()

    runprint = "Registration run " + str(runnr)
    if not verbose: runprint += ", moving = " + str(movingnr) + ", fixed = "+str(fixednr)+", slice "+str(slicenr)
    print(runprint)
    if verbose: print("------------------")
        
    selectslice(fixednr,slicenr)
    selectslice(movingnr,slicenr)

    fixednr = str(fixednr) + '_' + str(slicenr)
    movingnr = str(movingnr) + '_' + str(slicenr)

    #import images
    fixed_image_path = os.path.join(data_path,f"p{fixednr}\mr_bffe.mhd")
    moving_image_path = os.path.join(data_path,f"p{movingnr}\mr_bffe.mhd")

    #create model
    if not os.path.exists(elastix_path):
        raise IOError('Elastix cannot be found, please set the correct ELASTIX_PATH.')
    if not os.path.exists(transformix_path):
        raise IOError('Transformix cannot be found, please set the correct TRANSFORMIX_PATH.')
    
    output_dir = f'results{runnr}'
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)
    
    el = elastix.ElastixInterface(elastix_path=elastix_path)

    #run model
    el.register(fixed_image=fixed_image_path,
        moving_image=moving_image_path,
        parameters=[parameter_path],
        output_dir=output_dir,
        verbose=verbose)

    #writeoffset(moving_image_path, output_dir + r'\TransformParameters.0.txt')
    
    return runnr
    
def findnewrunnr():
    #Find the runnumber of the new file

    #Find all current files
    p = Path('./')
    list_of_files = list(p.glob('**'))
    
    newrunnr = 0
    for i, item in enumerate(list_of_files): #Loop over files
        if str(item)[:7] != 'results': continue #If file is result file
        try:
            runnr = int(str(item)[7:]) #Find runnumber
        except:
            continue
        if runnr > 99998: continue
        if runnr >= newrunnr: newrunnr = runnr + 1 #Define new runnumber
    return newrunnr

def writeoffset(data_file_path, parameter_path):
    #Read correct moving image offset
    with open(data_file_path,'r') as f:
        data_lines = f.readlines()
        
    for i, line in enumerate(data_lines):
        if line[0:6] == "Offset":
            offset = line[9:-1]
    
    #Write offset to parameter file
    with open(parameter_path,'r') as f:
        parameter_lines = f.readlines()
        
    with  open(parameter_path, 'w') as f:
        for i, line in enumerate(parameter_lines):
            if line[1:7] == "Origin":
                line = "(Origin " + offset + ")\n"
            f.writelines(line)
    
