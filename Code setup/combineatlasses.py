from transform import transform3d
from register import register3d
from pathlib import Path
from SimilarityMetrics import medpyDC, medpyHD, medpyRVD, findfixedmask3d, findtransformedmask
from weighted_average_tdc_threshold_updated import votingbased_DCweighted_3d_all, votingbased_DCweighted_3d_slice
import glob, os
import SimpleITK as sitk
from config import data_path
import shutil

nrslices = 86
runnr = 99999 #Make sure no old resultfiles are overwritten

def getaveragedicescore3d(moving,fixed,parameter_file):
    #Calculates the average dice score of registration
    #between all combinations of moving and fixed images    

    #moving = array with patientnumbers of moving images
    #fixed = array with patientnumbers of fixed images

    DCscore = 0
    n = 0
    
    for i in fixed:
        for j in moving:
            register3d(i,j,parameter_file,runnr=runnr,verbose=False)
            transform3d(j,runnr,transformmask=True)

            transformed = findtransformedmask(runnr)
            fixed = findfixedmask3d(i)

            DCscore += medpyDC(transformed,fixed)
            n += 1
            
    DCscore /= n
    
    output_dir = f'results{runnr}'
    shutil.rmtree(output_dir)

    return DCscore

def getaveragedicescore2d(moving,fixed,parameter_file):
    #Calculates the average dice score per slice of registration
    #between all combinations of moving and fixed images    

    #moving = array with patientnumbers of moving images
    #fixed = array with patientnumbers of fixed images

    DCscores = [0]*nrslices
    n = 0
    
    for i in fixed:
        for j in moving:
            register3d(i,j,parameter_file,runnr=runnr,verbose=False)
            transform3d(j,runnr,transformmask=True)

            transformed = findtransformedmask(runnr)
            fixed = findfixedmask3d(i)

            for k in range(nrslices):
                if sum(sum(x) for x in transformed[k])+sum(sum(x) for x in fixed[k]) == 0:
                    DCscores[k] += 1
                else:
                    DCscores[k] += medpyDC(transformed[k],fixed[k])
            n += 1
            
    for i in range(nrslices):
        DCscores[i] = DCscores[i]/n
    
    output_dir = f'results{runnr}'
    shutil.rmtree(output_dir)
    
    return DCscores

def IMatlasIFoptimize2d(atlasset,optimizeset,parameter_file):
    #This function performs step 2-5 of the plan,
    #with dice scores for each slice in the atlas images.
    DCscores = [[None]*nrslices]*len(atlasset)
    for image in range(len(DCscores)):
        DCscores[image]=getaveragedicescore2d([atlasset[image]],optimizeset,parameter_file)
    return DCscores

def IMatlasIFoptimize3d(atlasset,optimizeset,parameter_file):
    #This function performs step 2-5 of the plan,
    #with dice scores for each atlas image.
    DCscores = [None] * len(atlasset)
    for i in range(len(DCscores)):
        DCscores[i]=getaveragedicescore3d([atlasset[i]],optimizeset,parameter_file)
    return DCscores

def getfinalmasks2dDice(atlasset,validationset,DCscores2D,parameter_file, threshold1=0.5, threshold2=0.5):
    #This function performs step 6-8 + 10 of the
    #plan, using dice scores for each slice in the atlas images.
    fixed = validationset
    moving = atlasset

    finalmasks = []
    for i in fixed:
        transformedmasks = []
        for j in moving:
            register3d(i,j,parameter_file,runnr=runnr,verbose=False)
            transform3d(j,runnr,transformmask=True)

            transformedmasks.append(findtransformedmask(runnr))
        finalmasks.append(votingbased_DCweighted_3d_slice(transformedmasks, DCscores2D, threshold1, threshold2))
    
    output_dir = f'results{runnr}'
    shutil.rmtree(output_dir)

    return finalmasks

def getfinalmasks3dDice(atlasset,validationset,DCscore3D,parameter_file,threshold1=0.5,threshold2=0.5):
    #This function performs step 6-8 + 10 of the
    #plan, using dice scores for each atlas image.
    fixed = validationset
    moving = atlasset

    finalmasks = []
    for i in fixed:
        transformedmasks = []
        for j in moving:
            register3d(i,j,parameter_file,runnr=runnr,verbose=False)
            transform3d(j,runnr,transformmask=True)

            transformedmasks.append(findtransformedmask(runnr))
        finalmasks.append(votingbased_DCweighted_3d_all(transformedmasks, DCscore3D, threshold1, threshold2))
    
    output_dir = f'results{runnr}'
    shutil.rmtree(output_dir)

    return finalmasks

def validationscores(validationset,finalmasks):
    #This function performs step 9 of the plan.
    scores = []
    for i in range(len(validationset)):
        reference = findfixedmask3d(validationset[i])
        result = finalmasks[i]
        DCscore = medpyDC(result,reference)
        HDscore = medpyHD(result,reference)
        RVDscore = medpyRVD(result,reference)
        scores.append([DCscore,HDscore,RVDscore])
    return scores

def findnewmodelnr():
    #Find the modelrestultnumber of the new file

    #Find all current txt files
    os.chdir("./")
    list_of_files = list(glob.glob("*.txt"))
    
    newmodnr = 0
    for i, item in enumerate(list_of_files): #Loop over files
        if str(item)[:5] != 'model': continue #If file is model file
        try:
            modnr = int(str(item)[5:-4]) #Find modelnumber
        except:
            continue
        if modnr >= newmodnr: newmodnr = modnr + 1 #Define new modelnumber
    return newmodnr

def findnewmodelresultnr():
    #Find the runnumber of the new file

    #Find all current files
    p = Path('./')
    list_of_files = list(p.glob('**'))
    
    newrunnr = 0
    for item in list_of_files: #Loop over files
        if str(item)[:12] != 'modelresults': continue #If file is result file
        try:
            runnr = int(str(item)[12:]) #Find runnumber
        except:
            continue
        if runnr >= newrunnr: newrunnr = runnr + 1 #Define new runnumber
    return newrunnr

def write1dset(pset):
    pset_str = "["
    for i in range(len(pset)):
        if i == len(pset)-1: pset_str += str(pset[i]) + "]"
        else: pset_str += str(pset[i]) + ", "
    return pset_str

def writefullset(pset, name, dim):
    pset_str = name + " = "
    if dim == 1:
        pset_str += write1dset(pset)
    if dim == 2:
        seconddim_str = []
        for i in pset:
            seconddim_str.append(write1dset(i))
        pset_str += write1dset(seconddim_str)
    pset_str += "\n"
    return pset_str

def getlineafter(lines, keyword):
    for line in lines:
        if line[0:len(keyword)] == keyword:
            return line[len(keyword):]
    raise Exception("Keyword "+keyword+" not found.")
    
def read1dlist(line):
    array = []
    element = ""
    for item in line:
        if item == '[': continue
        if item == ']': break
        if item == " ": continue
        if item != ',': 
            element += item
        else:
            array.append(element)
            element = ""
    array.append(element)
    return array

def read2dlist(line):
    array = []
    starting_indices = []
    for i, item in enumerate(line):
        if item == '[': starting_indices.append(i)
    starting_indices.pop(0)
    for i in starting_indices:
        element = read1dlist(line[i:])
        array.append(element)
    return array

def listtofloat1d(l):
    for i in range(len(l)):
        l[i] = float(l[i])
    return l

def listtofloat2d(l):
    for i in range(len(l)):
        for j in range(len(l[i])):
            l[i][j] = float(l[i][j])
    return l

def listtoint1d(l):
    for i in range(len(l)):
        l[i] = int(l[i])
    return l

def listtoint2d(l):
    for i in range(len(l)):
        for j in range(len(l[i])):
            l[i][j] = int(l[i][j])
    return l
     
def writemodelfile(atlasset, optimizeset, parameter_file, DCscores, modelnr, threshold1, threshold2, sliceweighting):
    model_file = "model" + str(modelnr) + ".txt" 
    print("\nWriting model to " + model_file)
    with open(model_file, 'w') as f:
        f.writelines("These are the parameters of model "+str(modelnr)+".\n")
        
        if sliceweighting: f.writelines("The model applies weighting per slice.\n")
        else: f.writelines("The model applies weighting per image.\n")

        f.writelines(writefullset(atlasset,"atlasset",1))
        f.writelines(writefullset(optimizeset,"optimizeset",1))

        f.writelines("\nUsed parameterfile:\n")
        f.writelines("-------------------------------------------------------------------------\n")        
        
        with open(parameter_file,'r') as fp:
            parameter_lines = fp.readlines()            
        for line in parameter_lines:
            f.writelines(line)
            
        f.writelines("-------------------------------------------------------------------------\n")        
        
        f.writelines("Note that you can change the two thresholds below by hand. This won't 'break' anything.\nHowever, any evaluation results will not be correct anymore after changing the thresholds.\n")
        f.writelines("Threshold1 = "+str(threshold1)+'\n')
        f.writelines("Threshold2 = "+str(threshold2)+'\n')
        if sliceweighting: f.writelines(writefullset(DCscores,"Found weights",2))
        else: f.writelines(writefullset(DCscores,"Found weights",1))

def writevalidationfile(validationset, val_scores, modelnr):
    validation_file = "modelvalidation" + str(modelnr) + ".txt" 
    print("\nWriting validation results to " + validation_file)
    with open(validation_file, 'w') as f:
        f.writelines("These are the results of model"+str(modelnr)+".\n")
        f.writelines(writefullset(validationset,"validationset",1))

        f.writelines("Validation scores:\n")
        for i, item in enumerate(validationset):
            f.writelines("p" + str(item) + '\n')
            f.writelines("DC = " + str(val_scores[i][0]) + '\n')
            f.writelines("HD = " + str(val_scores[i][1]) + '\n')
            f.writelines("AVD = " + str(val_scores[i][2]) + '\n')

def readmodelfile(modelnr):
    model_file = "model" + str(modelnr) + ".txt"
    with open(model_file,'r') as f:
        model_lines = f.readlines()

    if getlineafter(model_lines,"The model applies weighting per ") == "image.\n": sliceweighting=False
    else: sliceweighting = True

    atlasset = listtoint1d(read1dlist(getlineafter(model_lines,"atlasset = ")))
    optimizeset = listtoint1d(read1dlist(getlineafter(model_lines,"optimizeset = ")))

    parameterlines = False
    firstline = False
    parameter_file = "parameters_to_execute_model.txt"
    with open(parameter_file, 'w') as fp:
        for line in model_lines:
            if parameterlines:
                if line[0:5] == "-----":
                    if firstline: break
                    firstline = True
                    continue
                fp.writelines(line)
            if line == "Used parameterfile:\n":
                parameterlines = True
        
    if sliceweighting: DCscores = listtofloat2d(read2dlist(getlineafter(model_lines,"Found weights = ")))
    else: DCscores = listtofloat1d(read1dlist(getlineafter(model_lines,"Found weights = ")))

    try:
        threshold1 = float(getlineafter(model_lines,"Threshold1 = "))
        threshold2 = float(getlineafter(model_lines,"Threshold2 = "))
    except Exception:
        updatemodelfile(modelnr)
        atlasset, optimizeset, parameter_file, DCscores, threshold1, threshold2, sliceweighting = readmodelfile(modelnr)
        print("If this line appears over and over again, there is a mistake in the model file! If it appears only once, don't worry.")
    return atlasset, optimizeset, parameter_file, DCscores, threshold1, threshold2, sliceweighting

def updatemodelfile(modelnr):
    model_file = "model" + str(modelnr) + ".txt"
    with open(model_file,'r') as f:
        model_lines = f.readlines()
   
    try:
        getlineafter(model_lines,"Threshold = ")
    except Exception:
        raise Exception("Model "+str(modelnr)+" probably already is in the new format, you don't need to run this function on it.")
   
    with open(model_file, 'w') as f:
        for line in model_lines:
            if line[:9] == "Threshold":
                f.writelines("Note that you can change the two thresholds below by hand. This won't 'break' anything.\nHowever, any evaluation results will not be correct anymore after changing the thresholds.\n")
                f.writelines("Threshold1 = "+getlineafter(model_lines,"Threshold = "))
                f.writelines("Threshold2 = 0.5\n")
            else:
                f.writelines(line)
        
def readvalidationfile(modelnr):
    validation_file = "modelvalidation" + str(modelnr) + ".txt" 
    with open(validation_file,'r') as f:
        validation_lines = f.readlines()

    validationset = listtoint1d(read1dlist(getlineafter(validation_lines,"validationset = ")))

    scorelines = False
    val_scores = []
    values = [None]*3
    for line in validation_lines:
        if line == "Validation scores:\n":
            scorelines = True
        if scorelines:
            if line[0:2] == "DC":
                values[0] = float(line[5:])
            elif line[0:2] == "HD":
                values[1] = float(line[5:])
            elif line[0:3] == "AVD":
                values[2] = float(line[6:])
                val_scores.append(values.copy())
    return validationset, val_scores

def createmodel2d(atlasset,optimizeset,parameter_file,threshold1=0.5,threshold2=0.5):
    modelnr = findnewmodelnr()
    print("Creating model " + str(modelnr) + '\n')

    DCscores2D = IMatlasIFoptimize2d(atlasset,optimizeset,parameter_file)   
    writemodelfile(atlasset, optimizeset, parameter_file, DCscores2D, modelnr, threshold1, threshold2, sliceweighting = True)    
    return modelnr

def createmodel3d(atlasset,optimizeset,parameter_file,threshold1=0.5, threshold2=0.5):
    modelnr = findnewmodelnr()
    print("Creating model " + str(modelnr) + '\n')

    DCscores2D = IMatlasIFoptimize3d(atlasset,optimizeset,parameter_file)   
    writemodelfile(atlasset, optimizeset, parameter_file, DCscores2D, modelnr, threshold1, threshold2, sliceweighting = False)
    return modelnr

def validatemodel(modelnr,validationset):
    print("Validating model " + str(modelnr) + '\n')
    atlasset, optimizeset, parameter_file, DCscores, threshold1, threshold2, sliceweighting = readmodelfile(modelnr)
    if sliceweighting:
        finalmasks = getfinalmasks2dDice(atlasset,validationset,DCscores,parameter_file,threshold1,threshold2)
        val_scores = validationscores(validationset,finalmasks)
    else:
        finalmasks = getfinalmasks3dDice(atlasset,validationset,DCscores,parameter_file,threshold1,threshold2)
        val_scores = validationscores(validationset,finalmasks)
    writevalidationfile(validationset, val_scores, modelnr)
    return val_scores

def runmodel(modelnr,unknownset):
    print("Running model " + str(modelnr) + '\n')
    atlasset, optimizeset, parameter_file, DCscores, threshold1, threshold2, sliceweighting = readmodelfile(modelnr)
    if sliceweighting:
        finalmasks = getfinalmasks2dDice(atlasset,unknownset,DCscores,parameter_file,threshold1,threshold2)
    else:
        finalmasks = getfinalmasks3dDice(atlasset,unknownset,DCscores,parameter_file,threshold1,threshold2)
        
    outputnr = findnewmodelresultnr()
    output_dir = f'modelresults{outputnr}'
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)

    print("Saving results in folder " + output_dir)    
    for i in range(len(unknownset)):
        filename = "maskp"+str(unknownset[i])+'.mhd'
        image = sitk.GetImageFromArray(finalmasks[i])
        sitk.WriteImage(image,os.path.join(output_dir, filename))
        
        original_file = os.path.join(data_path,f"p{unknownset[i]}\mr_bffe.mhd")
        with open(original_file,'r') as f:
            or_data_lines = f.readlines()
        with open(os.path.join(output_dir, filename),'r') as f:
            new_data_lines = f.readlines()
                
        with open(os.path.join(output_dir, filename), 'w') as f:
            for j, line in enumerate(new_data_lines):
                if line[:6] == "Offset":
                    for or_line in or_data_lines:
                        if or_line[:6] == "Offset":
                            f.writelines(or_line)
                elif line[:16] == "CenterOfRotation":
                    for or_line in or_data_lines:
                        if or_line[:16] == "CenterOfRotation":
                            f.writelines(or_line)
                elif line[:14] == "ElementSpacing":
                    for or_line in or_data_lines:
                        if or_line[:14] == "ElementSpacing":
                            f.writelines(or_line)       
                elif line[:21] == "AnatomicalOrientation":
                    for or_line in or_data_lines:
                        if or_line[:21] == "AnatomicalOrientation":
                            f.writelines(or_line)   
                else: f.writelines(line)

    with open(os.path.join(output_dir, "description.txt"), 'w') as f:
        f.writelines("This folder contains the results of model "+str(modelnr))
    return finalmasks