
import empymod
import numpy as np
import time
import multiprocessing as mp


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

HCP = empymod.loop(Hsource, Hreceivers, depth, res, freq, xdirect=None, mrec = 'loop')
VCP = empymod.loop(Vsource, Vreceivers, depth, res, freq, xdirect=None, mrec = 'loop')
PRP = empymod.loop(Hsource, Vreceivers, depth, res, freq, xdirect=None, mrec = 'loop')

Zdata = np.hstack((HCP, VCP, PRP))

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

# Array to store values

Zcube = np.zeros((nsl, nsl, nsl, nsl, nsl, 9), dtype = 'complex') # 9 coil geometries, 5 parameters

# Loop to create hypercube

startTime = time.time()

err = 1

def forward_parallel(i, err=1):

    is1 = i # change to nsl
    res[1] = 1/conds[is1] # set resistivity of first layer

    for is2 in range(0, nsl):
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

                    Zcube[is1, is2, is3, it1, it2, 0:3] = HCP
                    Zcube[is1, is2, is3, it1, it2, 3:6] = VCP
                    Zcube[is1, is2, is3, it1, it2, 6:9] = PRP

                    Z = np.hstack((HCP, VCP, PRP))

                    # Calculate amplitude of difference

                    nZdiff = np.abs(Z - Zdata) **2 / np.abs(Zdata)**2

                    merr = np.log10(np.sqrt(np.sum(nZdiff)))

                    if merr < err: # until error increases
                        # set model values
                        ms1 = is1
                        ms2 = is2
                        ms3 = is3
                        mt1 = it1
                        mt2 = it2
                        err = merr

    return Z, nZdiff, merr
    
# Start parallel process

pool_obj = mp.Pool()
Result = pool_obj.map(forward_parallel, range(0,nsl))

# End of process, store cubes

executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))

Result = np.asanyarray(Result, dtype=object)

#print('Best model is found at an error of ' + str(10**err) +' for')
#print('sigma_1 = '+ str(conds[ms1])+ ' S/m, d_1 ' + str(thicks[mt1]) + 'm')
#print('sigma_2 = '+ str(conds[ms2])+ ' S/m, d_2 ' + str(thicks[mt1] + thicks[mt2]) + 'm')
#print('sigma_3 = '+ str(conds[ms3])+ ' S/m')

np.save('Zcube51', Zcube)
np.save('Result51', Result)
