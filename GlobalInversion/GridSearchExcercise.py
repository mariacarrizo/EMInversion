# Grid search for EM Inversion using number of samples 51 per parameter

import empymod
import numpy as np
import matplotlib.pyplot as plt
import time

## Create a synthetic model

# Receivers geometry

offsets = np.array([2, 4, 8]) # in meters
dip = np.array([0, 90])

Hreceivers = [offsets, offsets*0, 0, 0, 0]
Vreceivers = [offsets, offsets*0, 0, 0, 90]

# Source geometry

Hsource = [0, 0, 0 ,0 , 0]
Vsource = [0, 0, 0, 0, 90]

# Frequency

freq = 9000

# parameters for the synthetic model

nlayer = 3 # number of layer
npos = 20 # number of sampling positions

resistivities = [10,50,10]
res = np.ones((npos, nlayer))*resistivities
x = np.linspace(0, 20, npos)[:,None]
thk1 = 2 + 0.2 * np.sin(x*np.pi*2) # wave
thk2 = 2 + np.sin(x*np.pi*2) # wave
depthmax = 10
ny = 50

# Define forward operator

def EMforward(x):
    # x is input model
    surface = np.array([0])
    thkx = np.array(x[0:2])
    depthx = np.hstack((surface, thkx[0], thkx[0]+thkx[1]))
    res_ground = np.array(x[2:])
    res_air = np.array([2e14])
    resx = np.hstack((res_air, res_ground))
    HCP = empymod.loop(Hsource, Hreceivers, depthx, resx, freq, xdirect=None, mrec = 'loop', verb=0)
    VCP = empymod.loop(Vsource, Vreceivers, depthx, resx, freq, xdirect=None, mrec = 'loop', verb=0)
    PRP = empymod.loop(Hsource, Vreceivers, depthx, resx, freq, xdirect=None, mrec = 'loop', verb=0)
    Z = np.hstack((HCP, VCP, PRP))
    return Z

# Load solutions hypercube

Zcube = np.load('Zcube51.npy')

# Inputs for inversion:

# Create empty array for true model in each position
model = []

# Create empty array for true data in each position
data = []

# Calculate data responses of the synthetic model
for i in range(npos):
    model_i = np.array([thk1[i][0], thk2[i][0]]+resistivities) # True model 
    model.append(model_i)
    data.append(EMforward(model_i)) # creating data

# Relative error array
error = 1e-2 # introduce here the error you want to test
relativeError = np.ones_like(data[0]) * error

nsl = 51 # sampling number of the cube

s0 = -2 # minimum conductivity in S/m
s1 = -0.8 # maximum conductivity in S/m
# conductivities array
conds = np.logspace(s0, s1, nsl)

th0 = 0.1 # minimum thickness in m
th1 = 5   # maximum thickness in m
# thickness array
thicks = np.linspace(th0, th1, nsl)

# Create gridsearch function

def gridsearch(Zdata):
    err = 1
    
    for is1 in range(nsl):
        for is2 in range(nsl):
            for is3 in range(nsl):
                for it1 in range(nsl):
                    for it2 in range(nsl):
                        Z = Zcube[is1, is2, is3, it1, it2, :]
                        nZdiff = np.abs(Z - Zdata) **2 / np.abs(Zdata)**2

                        merr = np.log10(np.sqrt(np.sum(nZdiff)))

                        if merr < err:
                            # set model values
                            ms1 = is1
                            ms2 = is2
                            ms3 = is3
                            mt1 = it1
                            mt2 = it2
                            err = merr
                            
    model = [thicks[mt1], thicks[mt2], 1/conds[ms1], 1/conds[ms2], 1/conds[ms3]]
    return model

# Start inversion

model_est_list =[] # empty array to store the model estimated

for i in range(npos):
    
    dataE = np.array(data[i]).copy()
    dataE *= np.random.randn(len(dataE)) * relativeError + 1.0

    model_est = gridsearch(dataE)
    
    model_est_list.append(model_est)
    
    print('Estimated position ', i)

# Model estimation array    
model_est_arr = np.array(model_est_list)

# Save estimated model
np.save('ModelEst20posZ51', model_est_arr)

