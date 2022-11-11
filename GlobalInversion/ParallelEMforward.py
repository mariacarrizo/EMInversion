import empymod
import numpy as np
import time
from joblib import Parallel, delayed

# Define Dualem-482 geometry

offsets = np.array([2, 4, 8]) # in meters
height = -0.10 # meters height From ground surface to center of coils
rad = 0.08 # Define radius of coil (8 cm)

# Source and receivers geometry

# For HCP
Hsource = [-rad, rad, -rad, rad, height, height]
Hreceivers = [offsets-rad, offsets+rad,
              np.ones(3)*-rad, np.ones(3)*rad, height, height]

# For VCP
Vsource = [0, 0, height, 90, 0]
Vreceivers = [offsets, offsets*0, height, 90, 0]

# For PRP
Psource = [0, 0, height]
Preceivers = [offsets, offsets*0, height]

# Frequency

freq = 9000

# True data

res = [2e14, 10, 20, 10]
depth = [0, 2, 5]

# sampling of depth and conductivities
nsl = 51

s0 = -2 # minimum conductivity in S/m
s1 = -0.8 # maximum conductivity in S/m
# conductivities array
conds = np.logspace(s0, s1, nsl)

th0 = 0.1 # minimum thickness in m
th1 = 5   # maximum thickness in m
# thickness array
thicks = np.linspace(th0, th1, nsl)

def forward_parallel(is1, is2, is3, it1, it2):
    time.sleep(0.01)
    res[1] = 1/conds[is1] # set resistivity of first layer
    res[2] = 1/conds[is2] # set resistivity of second layer
    res[3] = 1/conds[is3] # set resistivity of third layer
    depth[1] = thicks[it1] # set thickness of first layer
    depth[2] = depth[1] + thicks[it2] # set thickness of second layer

    # Compute fields

    HCP_Hs = empymod.loop(Hsource, Hreceivers, depth, res, freq, xdirect=None, mrec = 'loop', verb=0)
    VCP_Hs = empymod.loop(Vsource, Vreceivers, depth, res, freq, xdirect=None, mrec = 'loop', verb=0)
    PRP_Hs = empymod.dipole(Psource, Preceivers, depth, res, freq, ab=64, xdirect=None, verb=0)

    HCP_Hp = empymod.loop(Hsource, Hreceivers, depth=[], res=[2e14], freqtime=freq, mrec = 'loop', verb=0)
    VCP_Hp = empymod.loop(Vsource, Vreceivers, depth=[], res=[2e14], freqtime=freq, mrec = 'loop', verb=0)
    PRP_Hp = empymod.dipole(Psource, Preceivers, depth=[], res=[2e14], freqtime=freq, ab=66, verb=0)

    Q_HCP = np.abs(np.imag(HCP_Hs/HCP_Hp))
    Q_VCP = np.abs(np.imag(VCP_Hs/VCP_Hp))
    Q_PRP = np.abs(np.imag(PRP_Hs/PRP_Hp))

    Z = np.hstack((Q_HCP, Q_VCP, Q_PRP))

    return Z

starttime = time.time()

Results = Parallel(n_jobs=48,verbose=1)(delayed(forward_parallel)(i, j, k, m, n) for i in range(nsl) for j in range(nsl) 
                                         for k in range(nsl) for m in range(nsl) for n in range(nsl))

endtime = time.time() - starttime
print('Execution time parallel is:', endtime)

np.save('Results51', Results51)                     
