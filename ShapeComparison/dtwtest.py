import numpy as np
import matplotlib.pyplot as plt
from dtw import *
from curvetools import *
import scipy.io as sio
import glob
import time
from multiprocessing import Pool


## Step 1: Load in curves
NUM_PER_CLASS = 20
sigma = 4
N = 200

classes = ["apple","bat","beetle", "bird","Bone","bottle","brick","camel","car","carriage","cellular_phone","children","chopper","classic","comma","deer","device0","device1","device2","device8","elephant","face","fountain","Glas","hammer","HCircle","Heart","jar","key","lmfish","Misk","octopus","pencil","personal_car","rat","ray","sea_snake","shoe","spoon","spring","stef","teddy","tree","truck","watch"]

curves = np.zeros((len(classes)*NUM_PER_CLASS, N, 2))
idx = 0
for i, c in enumerate(classes):
    print(c)
    for j, filename in enumerate(glob.glob("mpeg7/{}*.gif".format(c))):
        X = get_curve(filename)
        X = X-np.mean(X, axis=0)
        # Downsample for efficiency
        s = get_arclen(get_curv_vectors(X, 0, sigma, loop=True)[1])
        X = arclen_resample(X, s, N)
        curves[idx, :, :] = X 
        idx += 1

def compute_row(args):
    (curves, i) = args
    row = np.zeros(N)
    for j in range(i+1, N):
        _, cost, _ = dtw_cyclic(get_csm(curves[i, :, :], curves[j, :, :]))
        row[j] = cost
    return row

## Step 2: Compute all pairwise distances
n_threads = 8

N = len(curves)
curves = curves[0:N, :, :]

pool = Pool(n_threads)
tic = time.time()
res = pool.map(compute_row, zip([curves]*N, range(N)))
D = np.array(res)
D = D + D.T
print("Elapsed Time: ", time.time()-tic)
sio.savemat("cyclicdtw.mat", {"D":D})