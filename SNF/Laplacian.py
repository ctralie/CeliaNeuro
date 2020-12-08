import sys
import scipy
import scipy.sparse as sparse
import numpy as np
import numpy.linalg as linalg
import scipy.linalg as sclinalg
from sklearn.cluster import KMeans


def getUnweightedLaplacianEigsDense(W):
    """
    Get eigenvectors of the unweighted Laplacian
    Parameters
    ----------
    W: ndarray(N, N)
        A symmetric similarity matrix that has nonnegative entries everywhere
    
    Returns
    -------
    v: ndarray(N, N)
        A matrix of eigenvectors
    """
    D = scipy.sparse.dia_matrix((W.sum(1).flatten(), 0), W.shape).toarray()
    L = D - W
    try:
        _, v = linalg.eigh(L)
    except:
        return np.zeros_like(W)
    return v

def getSymmetricLaplacianEigsDense(W):
    """
    Get eigenvectors of the weighted symmetric Laplacian
    Parameters
    ----------
    W: ndarray(N, N)
        A symmetric similarity matrix that has nonnegative entries everywhere
    
    Returns
    -------
    v: ndarray(N, N)
        A matrix of eigenvectors
    """
    D = scipy.sparse.dia_matrix((W.sum(1).flatten(), 0), W.shape).toarray()
    L = D - W
    SqrtD = np.sqrt(D)
    SqrtD[SqrtD == 0] = 1.0
    DInvSqrt = 1/SqrtD
    LSym = DInvSqrt.dot(L.dot(DInvSqrt))
    try:
        _, v = linalg.eigh(LSym)
    except:
        return np.zeros_like(W)
    return v

def getRandomWalkLaplacianEigsDense(W):
    """
    Get eigenvectors of the random walk Laplacian by solving
    the generalized eigenvalue problem
    L*u = lam*D*u
    Parameters
    ----------
    W: ndarray(N, N)
        A symmetric similarity matrix that has nonnegative entries everywhere
    
    Returns
    -------
    v: ndarray(N, N)
        A matrix of eigenvectors
    """
    D = scipy.sparse.dia_matrix((W.sum(1).flatten(), 0), W.shape).toarray()
    L = D - W
    try:
        _, v = sclinalg.eigh(L, D)
    except:
        return np.zeros_like(W)
    return v

def spectralCluster(v, dim, rownorm=False):
    """
    Given Laplacian eigenvectors associated with a graph, perform 
    spectral clustering
    Parameters
    ----------
    v: ndarray(N, k)
        A matrix of eigenvectors, excluding the zeroeth
    dim: int
        Dimension of spectral clustering, <= k
    rownorm: boolean
        Whether to normalize each row (if using symmetric Laplacian)

    Returns
    -------
    labels: ndarray(N)
        Cluster membership for each point
    """
    assert dim <= v.shape[1]
    x = np.array(v[:, 0:dim])
    if rownorm:
        norms = np.sqrt(np.sum(x**2, 1))
        norms[norms == 0] = 1
        x /= norms[:, None]
    return KMeans(n_clusters = dim, n_init=100, max_iter=500).fit(x).labels_