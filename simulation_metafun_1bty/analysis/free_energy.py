import numpy as np
import math
import plumed
import mdtraj as mdj
import matplotlib.pyplot as plt

# finds the relative free energy of a 1D FES, usually
# of the proj CV
# you will see where this is used in the cell below
def fes_rel(fes_list, fes_all=None, kT=2.5):
    results = []
    for i in fes_list:
        temp = 0
        for j in i:
            temp += math.exp(-j / kT)
        results.append(temp)
    total = 0
    if fes_all is None:
        for i in results:
            total += i
    else:
        for i in fes_all:
            total += i

    results_2 = []
    for i in results:
        if i > 0:
            results_2.append(-kT * math.log(i / total))
        else:
            results_2.append(np.inf)
    min_res = min(results_2)
    return [i - min_res for i in results_2]

funnel_correction = 2.45
fes_data = np.loadtxt("fes_1d.dat")
fes_1d = fes_rel(fes_data)
print(len(fes_1d))