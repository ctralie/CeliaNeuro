import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import glob
from SimilarityFusion import *

def plot_average(Ws):
    Im =fused_score(Ws)
    Im_Disp = np.array(Im)
    np.fill_diagonal(Im_Disp, 0)
    plt.imshow(Im_Disp, interpolation = 'none', cmap = 'magma_r')

def plot_simple_average(Ws):
    plt.figure(figsize=(8, 8))
    plot_average(Ws)
    plt.title("Simple Average")
    plt.savefig("SimpleAverage.png", dpi=300, bbox_inches='tight')

def make_healthy(Ks):
    files = glob.glob("Healthy/*.mat")
    Ws = []
    for f in files:
        s = f.split("_")[0].split("/")[-1]
        Ws.append(sio.loadmat(f)[s])
    plot_simple_average(Ws)
    sio.savemat("SimpleAverage.mat", {"W":fused_score(Ws)})
    plt.figure(figsize=(8, 8))
    for K in Ks:
        plt.clf()
        Im = snf_ws(Ws, K = K, niters = 20, reg_diag = True)
        Im_Disp = np.array(Im)
        np.fill_diagonal(Im_Disp, 0)
        plt.imshow(Im_Disp, interpolation = 'none', cmap = 'magma_r')
        plt.title("K = {}".format(K))
        plt.savefig("SNF_K{}.png".format(K), dpi=300, bbox_inches='tight')
        sio.savemat("SNF_K{}.mat".format(K), {"W":Im})

make_healthy([3, 5, 10, 15, 20, 25])