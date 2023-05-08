
import yaml
import numpy as np
import matplotlib.pyplot as plt

import argparse

filename = "debug_file.yaml"
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", default=filename)
args = parser.parse_args()
filename = args.file



with open(filename,"r") as f:
    data = yaml.safe_load(f)



D = {}


# max_time = 0

for d in data:
    name  = d["name"] 
    if name not in D:
        D[name] = [d]
    else: 
        D[name] += [d]
    # if d.time > max_time
    #     max_time = d.time

# now i have grouped by names

for k,v in D.items():
    times  = []
    costs = []
    for vv in v: 
        times.append(vv["time"])
        costs.append(vv["cost"])
    print("k" , k)
    plt.plot( times, costs , 'o',  label=k)
plt.legend()
plt.show()

    

