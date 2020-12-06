import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import sparse
import glob
import os
import nilearn
import nilearn.masking
import nilearn.image
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

def do_healthy_fusion(Ks):
    """
    Fuse healthy subjects
    """
    files = glob.glob("Normalized/Controls_for_reference_graph/*.mat")
    Ws = []
    for f in files:
        s = f.split("_")[-2].split("/")[-1]
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

def normalize_by_atlas():
    afolder = "3_Atlases_ModeDenoising"
    folders = ["Controls_for_reference_graph", "Controls_JHU", "Patients_BU", "Patients_JHU", "Patients_NU"]
    for folder in folders:
        folderpath = "Normalized/{}".format(folder)
        if not os.path.exists(folderpath):
            os.mkdir(folderpath)
        for f in glob.glob("{}/*.mat".format(folder)):
            ## Step 1: Load in atlas
            s = f.split("_")[-2].split("/")[-1]
            print(s)
            W = sio.loadmat(f)[s]
            ## Step 2: Load in labeled parcels
            path = "{}/{}*.nii".format(afolder, s)
            atlasfile = glob.glob(path)[0]
            img = nilearn.image.load_img(atlasfile)
            ## Step 3: Count the parcels of each type
            # Use sparse matrix to quickly create counts
            I = img.get_fdata().flatten()
            I = np.array(I, dtype=int)
            J = np.zeros_like(I)
            V = np.ones_like(I)
            counts = sparse.coo_matrix((V, (I, J)), (np.max(I)+1, 1))
            counts = counts.toarray().flatten()[1::]
            ## Step 4: Normalize entries of the matrix by dividing each
            ## element by the minimum count of the two parcels it connects
            norm = np.minimum(counts[:, None], counts[None, :])
            norm[norm == 0] = 1
            W /= norm
            ## Step 5: Save the results
            filepath="{}/{}".format(folderpath, f.split("/")[-1])
            sio.savemat(filepath, {s:W})

            

        
#normalize_by_atlas()
do_healthy_fusion([3, 5, 10, 15, 20, 25])