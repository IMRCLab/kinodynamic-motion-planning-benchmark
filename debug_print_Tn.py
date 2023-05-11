


import matplotlib.pyplot as plt


filename = "data_Tn.txt"


data = []

with open(filename) as f:

    for line in f:
        ds = line.strip().split()
        di = [float(x) for x in ds ]
        data.append(di)


Xs = [ d[0] for d in data] 
Ys = [ d[1] for d in data] 
Thetas = [ d[2] for d in data] 





plt.plot( Xs,Ys , '.')
plt.show()
