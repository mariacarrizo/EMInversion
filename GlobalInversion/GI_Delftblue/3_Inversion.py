### Code that creates searches in Lookup table for the corresponding model to the data
### 1D 3 layered model

## Import libraries

import empymod
import numpy as np
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed

# sampling of depth and conductivities in the table
nsl = 51

s0 = -3 # minimum conductivity in S/m
s1 = -0.5 # maximum conductivity in S/m
# conductivities array
conds = np.logspace(s0, s1, nsl)

th0 = 0.1 # minimum thickness in m
th1 = 7   # maximum thickness in m
# thickness array
thicks = np.linspace(th0, th1, nsl)

## Load lookup table

exact = np.load('LU_Table.npy')

## Define functions to search solution in look up table

def gridsearch(data):
    err = 1
    indx=0
    for i in range(np.shape(exact)[0]):
        nZdiff = np.abs(exact[i] - data) **2 / np.abs(data)**2
        merr = np.log(np.sqrt(np.sum(nZdiff)))
        if merr < err:
            indx = i
            err = merr.copy()
    return indx

def invert(index):
    for i in range(len(conds)):
        for j in range(len(conds)):
            for k in range(len(conds)):
                for m in range(len(thicks)):
                    for n in range(len(thicks)):
                        idx = n + m*nsl + k*nsl**2 + j*nsl**3 + i*nsl**4
                        if index == idx:
                            model = np.array([conds[i], conds[j], conds[k], thicks[m], thicks[n]])
                            return model

## Load true synthetic model and data

model = np.load('model_synth.npy')
data = np.load('data_synth.npy')

## Start inversion

model_est_list =[] # empty array to store the model estimated

for i in range(npos):   
    data = np.array(data[i]).copy()
    # For each position search solution for data
    model_est_i = invert(gridsearch(data))
    model_est_list.append(model_est)

# Model estimation array  
model_est = np.array(model_est_list)

## Save model estimated

np.save('model_est', model_est)