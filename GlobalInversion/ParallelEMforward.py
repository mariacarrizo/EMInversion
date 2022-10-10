import empymod
import numpy as np
import time
from joblib import Parallel, delayed

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

# True data

res = [2e14, 10, 20, 10]
depth = [0, 2, 5]

# sampling of depth and conductivities
nsl = 5

s0 = -2 # minimum conductivity in S/m
s1 = -0.8 # maximum conductivity in S/m
# conductivities array
conds = np.logspace(s0, s1, nsl)

th0 = 0.1 # minimum thickness in m
th1 = 5   # maximum thickness in m
# thickness array
thicks = np.linspace(th0, th1, nsl)

def forward_parallel(is1, is2):
    time.sleep(1)
    res[1] = 1/conds[is1] # set resistivity of first layer
    res[2] = 1/conds[is2] # set resistivity of second layer
        
    for is3 in range(0, nsl): 
        res[3] = 1/conds[is3] # set resistivity of third layer
          
        for it1 in range(0, nsl):
            depth[1] = thicks[it1] # set thickness of first layer
                
            for it2 in range(0, nsl):
                depth[2] = depth[1] + thicks[it2] # set thickness of second layer
                    
                # Compute fields
                   
                HCP = empymod.loop(Hsource, Hreceivers, depth, res, freq, xdirect=None, mrec = 'loop', verb = 0)
                VCP = empymod.loop(Vsource, Vreceivers, depth, res, freq, xdirect=None, mrec = 'loop', verb = 0)
                PRP = empymod.loop(Hsource, Vreceivers, depth, res, freq, xdirect=None, mrec = 'loop', verb = 0)
                    
                # Store in hypercube
                    
               # Zcube[is1, is2, is3, it1, it2, 0:3] = HCP
               # Zcube[is1, is2, is3, it1, it2, 3:6] = VCP
               # Zcube[is1, is2, is3, it1, it2, 6:9] = PRP
                    
                Z = np.hstack((HCP, VCP, PRP))
                    
                # Calculate amplitude of difference
                    
                nZdiff = np.abs(Z - Zdata) **2 / np.abs(Zdata)**2

                merr = np.log10(np.sqrt(np.sum(nZdiff)))
                    
    return Z, nZdiff, merr

starttime = time.time()
Results = Parallel(n_jobs=48)(delayed(forward_parallel)(i, j) for i in range(nsl) for j in range(nsl))

endtime = time.time() - starttime
print('Execution time parallel is:', endtime)