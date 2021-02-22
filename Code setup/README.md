How to use the registration code

1. Download the code. (config.py, main.py, register.py, single_slice.py, transform.py, visualize.py, paramter file)
2. Change the paths in config.py to the paths corresponding with the location files on your computer.
3. Go to main.py and write down the wanted parameters (movingnr, fixednr, slicenr (in 2D case) and parameter_file).
  - Do a registration:
    1. Uncomment the registration method, and run (2D or 3D).    
  - Do a transformation
    1. This can be done in the same run as doing a registration, in that case the runnr variable can be commented out.
    2. This can be done in a different run the doing the registration, in that case the runnr variable should conatain the correct runnumber.
    3. Uncomment the wanted transform line (2D or 3D), and run.
  - Do a visualisation:
    1. Uncomment the registration method, and run (2D or 3D).
    Note that Giulia uploaded another visualisation method (scroll_and_masks.py), which also works nicely.
    
    
    Some recent updates have been done to Optimizeparameters.py/ changeparameters.py/ transform.py/ SimilarityMetrics.py on 22/02. 
