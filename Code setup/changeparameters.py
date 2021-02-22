def changetxt(file, oldstring, newstring):
    txtfile = open(file, "rt")

    parameters = txtfile.read()
    newparameters = parameters.replace(oldstring, newstring)

    txtfile.close()

    newtxtfile = open(file, "wt")
    newtxtfile.write(newparameters)

    newtxtfile.close()

def changetxttuples(tuple):
    txtfile = open(tuple[0], "rt")

    parameters = txtfile.read()
    newparameters = parameters.replace(tuple[1], tuple[2])

    txtfile.close()

    newtxtfile = open(tuple[0], "wt")
    newtxtfile.write(newparameters)

    newtxtfile.close()

def replace(parameter_file,keyword,value):
    with open(parameter_file,'r') as f:
        parameter_lines = f.readlines()
            
    with open(parameter_file, 'w') as f:
        for i, line in enumerate(parameter_lines):
            print(line[1:len(keyword)])
            if line[1:len(keyword)+1] == keyword:
                line = "(" + keyword + " "+ str(value) + ")\n"
            f.writelines(line)
            
def readparameter(parameter_file,keyword):
    #read lines
    with open(parameter_file,'r') as f:
        parameter_lines = f.readlines()

    #return 
    for line in parameter_lines:
        if line[1:len(keyword)+1] == keyword:
            return(line[len(keyword)+2:-2])
    
    raise Exception("ERROR: parameter "+keyword+" not found in "+parameter_file)