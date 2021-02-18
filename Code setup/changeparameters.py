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
