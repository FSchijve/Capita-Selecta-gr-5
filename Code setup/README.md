How to use the registration code

1. Download the code. (config.py, main.py, register.py, single_slice.py, transform.py, visualize.py, paramter file)
2. Change the paths in config.py to the paths corresponding with the location files on your computer.
3. Go to main.py and write down the wanted parameters (movingnr, fixednr, slicenr (in 2D case) and parameter_file).
  - Do a registration:
    1. Uncomment the registration method, and run (2D or 3D).
    
- There is a mistake somewhere, so at this stage there are a few extra manual steps needed:
  1. Open the mr_bffe.mhd file of the moving image. Note that when using the 2D registration method, this is the file in the data folder: patientnumber_slicenumber
  2. Copy the Offset (for example, with patient 102, this offset is -46.9093 -77.4416 -67.0081)
  3. Open the TransformParameters.0.txt file, you can find this in the results folder.
  4. Paste this Offset behind "Origin".    
    
- Do a transformation
  1. This can be done in the same run as doing a registration, in that case the runnr variable can be commented out.
  2. This can be done in a different run the doing the registration, in that case the runnr variable should conatain the correct runnr.
  3. Uncomment the transform line, and run.
- Do a visualisation:
  1. Uncomment the registration method, and run (2D or 3D).
  
  Note that Giulia uploaded another visualisation method. It is better to use that one! 
