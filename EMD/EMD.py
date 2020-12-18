"""
Purpose: To compute all pairs Wasserstein on 1D distributions of periodicity
scores for each cell for each gray level.
Also clustering on this
"""
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from scipy import integrate


def getWassersteinPairs(X, dim = 30):
    """
    Compute all pairs of Wasserstein distances between 
    distributions of a list of values, assuming that
    all of the lists have the same number of values so that
    the CDF is easily invertible
    Parameters
    ----------
    X: ndarray(N, M)
        N lists, each with M unordered samples
    Returns
    -------
    D: ndarray(N, N)
        All pairs 1D Wasserstein distances between lists in X
    """
    bins = np.linspace(0, 1, dim)
    N = X.shape[0]
    D = np.zeros((N, N))
    for i in range(N):
        if i%250 == 0:
            print("%i of %i"%(i, N))
        histi, _ = np.histogram(X[i, :], bins=bins)
        for j in range(i+1, N):
            histj, _ = np.histogram(X[j, :], bins=bins)
            D[i, j] = wasserstein_distance(histi, histj)
    D += D.T
    return D


def getAUROC(x, y, do_plot=False, ChunkSize = 1000, MaxLevels = 10000):
    """
    Parameters
    ----------
    x: ndarray(N1)
        True samples
    y: ndarray(N2)
        False samples
    ChunkSize: int
        Size of chunks of levels to do at a time
    
    Returns
    -------
    {
        'TP': ndarray(N)
            True positive thresholds,
        'FP': ndarray(N)
            False positive thresholds,
        'auroc': float
            The area under the ROC curve
    }
    """
    x = np.sort(x)
    y = np.sort(y)
    levels = np.sort(np.unique(np.concatenate((x, y))))
    if len(levels) > MaxLevels:
        levels = np.sort(levels[np.random.permutation(len(levels))[0:MaxLevels]])
    N = len(levels)
    FP = np.zeros(N)
    TP = np.zeros(N)
    i = 0
    while i < N:
        idxs = i + np.arange(ChunkSize)
        idxs = idxs[idxs < N]
        ls = levels[idxs]
        FP[idxs] = np.sum(ls[:, None] < y[None, :], 1) / y.size
        TP[idxs] = np.sum(ls[:, None] < x[None, :], 1) / x.size
        i += ChunkSize
    idx = np.argsort(FP)
    FP = FP[idx]
    TP = TP[idx]
    auroc = integrate.trapz(TP, FP)
    if do_plot:
        plt.plot(FP, TP)
        plt.xlabel("False Positives")
        plt.ylabel("True Positives")
        plt.title("AUROC = %.3g"%auroc)
    return {'FP':FP, 'TP':TP, 'auroc':auroc}
