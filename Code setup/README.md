How to use the registration code

1. Download the code. (config.py, main.py, register.py, single_slice.py, transform.py, visualize.py, paramter file)
2. Change the paths in config.py to the paths corresponding with the location files on your computer.
3. Go to main.py and write down the wanted parameters (movingnr, fixednr, slicenr (in 2D case) and parameter_file).
  Do a registration:
    a. Uncomment the registration method, and run (2D or 3D).
    
There is a mistake somewhere, so at this stage there are a few extra manual steps needed:
  a. Open the mr_bffe.mhd file of the moving image. Note that when using the 2D registration method, this is the file in the data folder: patientnumber_slicenumber
  b. Copy the Offset (for example, with patient 102, this offset is -46.9093 -77.4416 -67.0081)
  c. Open the TransformParameters.0.txt file, you can find this in the results folder.
  d. Paste this Offset behind "Origin".    
    
  Do a transformation:
    a. This can be done in the same run as doing a registration, in that case the runnr variable can be commented out.
    b. This can be done in a different run the doing the registration, in that case the runnr variable should conatain the correct runnr.
    c. Uncomment the transform line, and run.
  Do a visualisation:
    a. Uncomment the registration method, and run (2D or 3D).
    
    Note that Giulia uploaded another visualisation method. It is better to use that one! 
