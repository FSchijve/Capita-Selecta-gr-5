How to use the code:

1. Download the code.
2. Change the paths in config.py to the paths corresponding with the location files on your computer.
3. Go to main.py and write down the wanted parameters (movingnr, fixednr, slicenr (in 2D case) and parameter_file).
  - Do a single registration:
    1. Uncomment the registration method, and run (2D or 3D). If you set verbose = False you get less annoying messages from elastix.
  - Do a single transformation
    1. This can be done in the same run as doing a registration, in that case the runnr variable can be commented out.
    2. This can be done in a different run than doing the registration, in that case the runnr variable should contain the correct runnumber.
    3. Uncomment the wanted transform line (2D or 3D), and run.
  - Do a single slice visualisation:
    1. Uncomment the registration method, and run (2D or 3D).
    Note that Giulia uploaded another visualisation method (scroll_and_masks.py), which also works nicely.
  - Create a model, following the steps in the "Dice score" tab of onenote:
    1. Create 3 lists with patientnumbers: an list with future atlasses, a list with images used for optimization and a list with images used for validation.
    2. Uncomment the model creation method (using weighting based on whole images (3d) or using weighting based on slices (2d)), and run.
    3. After creating a model, a model.txt file is created, containing the data of the model.
  - Validate a model
    1. This can be done in the same run as creating the model, in that case the modelnr variable can be commented out.
    2. This can be done in a different run than creating the model, in that case the modelnr variable should contain the correct modelnr.
    3. Uncomment the validation line (validatemodel(modelnr,validationset)), and run.
    4. After validating the model, a modelvalidation folder is created, containing validationresults.txt with the validation scores for the different validation images and the estimated masks of the validation set.
  - Run the model on a new dataset
    1. This can be done in the same run as creating the model, in that case the modelnr variable can be commented out.
    2. This can be done in a different run than creating the model, in that case the modelnr variable should contain the correcnt modelnr.
    3. Uncomment the runmodel line (runmodel(modelnr,unknownset)), and run. The unknownset are the numbers of the patients in the unknown set. These patients should be in the data folder, with the same data format as the original data.
    4. After running the model, a modelresults folder is created, containing the estimated masks of the unknown set.