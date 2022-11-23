### Code that creates synthetic model and synthetic data for a 1D 3 layered model

## Import libraries

import empymod
import numpy as np
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed

## Survey geometry

# Define Dualem-482 geometry
offsets = np.array([2, 4, 8]) # in meters
height = -0.10 # meters height From ground surface to center of coils
rad = 0.08 # Define radius of coil (8 cm)

# Source and receivers geometry

# For HCP
Hsource = [-rad, rad, -rad, rad, height, height]
Hreceivers = [offsets-rad, offsets+rad, np.ones(3)*-rad, np.ones(3)*rad, height, height]

# For VCP
Vsource = [0, 0, height, 90, 0]
Vreceivers = [offsets, offsets*0, height, 90, 0]

# For PRP
Psource = [0, 0, height]
Preceivers = [offsets, offsets*0, height]

# Frequency
freq = 9000

## Define forward function

def EMforward(x):
    # x is input model
    surface = np.array([0])
    thkx = np.array(x[0:2])
    depthx = np.hstack((surface, thkx[0], thkx[0]+thkx[1]))
    res_ground = np.array(x[2:])
    res_air = np.array([2e14])
    resx = np.hstack((res_air, res_ground))
    
    HCP_Hs = -empymod.loop(Hsource, Hreceivers, depthx, resx, freq, xdirect=None, mrec = 'loop',verb=0)
    VCP_Hs = empymod.loop(Vsource, Vreceivers, depthx, resx, freq, xdirect=None, mrec = 'loop', verb=0)
    PRP_Hs = empymod.dipole(Psource, Preceivers, depthx, resx, freq, ab=64, xdirect=None, verb=0)

    HCP_Hp = empymod.loop(Hsource, Hreceivers, depth=[], res=[2e14], freqtime=freq, mrec = 'loop', verb=0)
    VCP_Hp = empymod.loop(Vsource, Vreceivers, depth=[], res=[2e14], freqtime=freq, mrec = 'loop', verb=0)
    PRP_Hp = empymod.dipole(Psource, Preceivers, depth=[], res=[2e14], freqtime=freq, ab=66, verb=0)

    Q_HCP = np.imag(HCP_Hs/HCP_Hp)
    Q_VCP = np.imag(VCP_Hs/VCP_Hp)
    Q_PRP = np.imag(PRP_Hs/PRP_Hp)
    
    P_HCP = np.real(HCP_Hs/HCP_Hp)
    P_VCP = np.real(VCP_Hs/VCP_Hp)
    P_PRP = np.real(PRP_Hs/PRP_Hp)   
    
    return np.hstack((Q_HCP, Q_VCP, Q_PRP, P_HCP, P_VCP, P_PRP))

## Define parameters for the synthetic model

nlayer = 3 # number of layer
npos = 20 # number of sampling positions

resistivities = [10,50,10] # 3 layered resistivities
res = np.ones((npos, nlayer))*resistivities
x = np.linspace(0, 20, npos)[:,None]
thk1 = 2 + 0.2 * np.sin(x*np.pi*2) # wave
thk2 = 2 + np.sin(x*np.pi*2) # wave

## Inputs for inversion:

# Create empty array for true model in each position
model = []

# Create empty array for true data in each position
data = []

for i in range(npos):
    model_i = np.array([thk1[i][0], thk2[i][0]]+resistivities) # True model 
    model.append(model_i)
    data.append(EMforward(model_i)) # creating data
    
## Calculate normalization

data = np.array(data)

# data shape is (npos, geometries * 2) 

Q_HCP_2 = data[:,0]
Q_HCP_4 = data[:,1]
Q_HCP_8 = data[:,2]
Q_VCP_2 = data[:,3]
Q_VCP_4 = data[:,4]
Q_VCP_8 = data[:,5]
Q_PRP_2 = data[:,6]
Q_PRP_4 = data[:,7]
Q_PRP_8 = data[:,8]

P_HCP_2 = data[:,9]
P_HCP_4 = data[:,10]
P_HCP_8 = data[:,11]
P_VCP_2 = data[:,12]
P_VCP_4 = data[:,13]
P_VCP_8 = data[:,14]
P_PRP_2 = data[:,15]
P_PRP_4 = data[:,16]
P_PRP_8 = data[:,17]

# Calculate L2 norm

L2_Q_HCP_2 = np.sqrt(np.sum(Q_HCP_2**2))
L2_Q_HCP_4 = np.sqrt(np.sum(Q_HCP_4**2))
L2_Q_HCP_8 = np.sqrt(np.sum(Q_HCP_8**2))

L2_Q_VCP_2 = np.sqrt(np.sum(Q_VCP_2**2))
L2_Q_VCP_4 = np.sqrt(np.sum(Q_VCP_4**2))
L2_Q_VCP_8 = np.sqrt(np.sum(Q_VCP_8**2))

L2_Q_PRP_2 = np.sqrt(np.sum(Q_PRP_2**2))
L2_Q_PRP_4 = np.sqrt(np.sum(Q_PRP_4**2))
L2_Q_PRP_8 = np.sqrt(np.sum(Q_PRP_8**2))

L2_P_HCP_2 = np.sqrt(np.sum(P_HCP_2**2))
L2_P_HCP_4 = np.sqrt(np.sum(P_HCP_4**2))
L2_P_HCP_8 = np.sqrt(np.sum(P_HCP_8**2))

L2_P_VCP_2 = np.sqrt(np.sum(P_VCP_2**2))
L2_P_VCP_4 = np.sqrt(np.sum(P_VCP_4**2))
L2_P_VCP_8 = np.sqrt(np.sum(P_VCP_8**2))

L2_P_PRP_2 = np.sqrt(np.sum(P_PRP_2**2))
L2_P_PRP_4 = np.sqrt(np.sum(P_PRP_4**2))
L2_P_PRP_8 = np.sqrt(np.sum(P_PRP_8**2))

L2_Q_HCP = np.hstack((L2_Q_HCP_2, L2_Q_HCP_4, L2_Q_HCP_8))
L2_Q_VCP = np.hstack((L2_Q_VCP_2, L2_Q_VCP_4, L2_Q_VCP_8))
L2_Q_PRP = np.hstack((L2_Q_PRP_2, L2_Q_PRP_4, L2_Q_PRP_8))

L2_P_HCP = np.hstack((L2_P_HCP_2, L2_P_HCP_4, L2_P_HCP_8))
L2_P_VCP = np.hstack((L2_P_VCP_2, L2_P_VCP_4, L2_P_VCP_8))
L2_P_PRP = np.hstack((L2_P_PRP_2, L2_P_PRP_4, L2_P_PRP_8))

L2 = np.stack((L2_Q_HCP, L2_Q_VCP, L2_Q_PRP, L2_P_HCP, L2_P_VCP, L2_P_PRP))

#Store results
np.save('model_synth', model)
np.save('data_synth', data)
np.save('L2', L2)