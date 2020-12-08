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
from Laplacian import *


def normalize_by_atlas():
    afolder = "3_Atlases_ModeDenoising"
    folders = ["Controls_for_reference_graph", "Controls_JHU", "Patients_BU", "Patients_JHU", "Patients_NU"]
    plt.figure(figsize=(8, 8))
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
            #norm = counts[:, None]*counts[None, :]
            norm[norm == 0] = 1
            W /= norm
            ## Step 5: Save the results
            filepath="{}/{}".format(folderpath, f.split("/")[-1])
            sio.savemat(filepath, {s:W})
            plt.clf()
            plt.imshow(W, interpolation='none', cmap='magma_r')
            plt.title(f)
            plt.savefig("{}.png".format(s))

            


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

def do_healthy_fusion(Ks, do_plot = False):
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
    results = {}
    for K in Ks:
        Im = snf_ws(Ws, K = K, niters = 20, reg_diag = True)
        if do_plot:
            plt.clf()
            Im_Disp = np.array(Im)
            np.fill_diagonal(Im_Disp, 0)
            plt.imshow(Im_Disp, interpolation = 'none', cmap = 'magma_r')
            plt.title("K = {}".format(K))
            plt.savefig("SNF_K{}.png".format(K), dpi=300, bbox_inches='tight')
        results[K] = Im
        sio.savemat("SNF_K{}.mat".format(K), {"W":Im})
    return results

def spectral_cluster_w(W, max_clusters):
    V = getRandomWalkLaplacianEigsDense(W)[:, 1::]
    specfn = lambda v, dim: spectralCluster(v, dim, rownorm=False)
    labels = [specfn(V, neigs) for neigs in range(1, max_clusters+1)]
    labels = np.array(labels)
    return labels

def print_labels(labels):
    fin = open("AAL3v1_LabelNames_EvenLeft.txt")
    names = [n.split()[-1] for n in fin.readlines()]
    idxs = np.arange(labels.shape[1])
    for i in range(1, labels.shape[0]):
        fout = open("{}.txt".format(i+1), "w")
        for k in range(i+1):
            fout.write("Cluster {}\n".format(k+1))
            names_k = [names[idx] for idx in idxs[labels[i, :] == k]]
            for n in names_k:
                fout.write("{}\n".format(n))
            fout.write("\n\n")
        fout.close()


def plot_labels(W, labels):
    Im_Disp = np.array(W)
    np.fill_diagonal(Im_Disp, 0)
    plt.figure(figsize=(10, 10))
    for i in range(1, labels.shape[0]):
        cmap = 'tab10'
        if i > 9:
            cmap = 'tab20'
        L = labels[i, :]
        plt.clf()
        plt.subplot2grid((10, 10), (1, 1), rowspan=9, colspan=9)
        plt.imshow(Im_Disp, interpolation = 'none', cmap = 'magma_r')
        plt.axis('off')
        plt.subplot2grid((10, 10), (0, 1), rowspan=1, colspan=9)
        plt.imshow(L[None, :], cmap=cmap, interpolation = 'none', aspect='auto')
        plt.axis('off')
        plt.subplot2grid((10, 10), (1, 0), rowspan=9, colspan=1)
        plt.imshow(L[:, None], cmap=cmap, interpolation = 'none', aspect='auto')
        plt.axis('off')
        plt.savefig("Clusters{}.png".format(i+1), bbox_inches='tight')

#normalize_by_atlas()
fused = do_healthy_fusion([15]) #[3, 5, 10, 15, 20, 25]
W = fused[15]
labels = spectral_cluster_w(W, 20)
sio.savemat("SpectralLabels.mat", {"W":W})
print_labels(labels)
plot_labels(W, labels)