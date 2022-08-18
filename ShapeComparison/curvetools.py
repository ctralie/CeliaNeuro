import numpy as np
import matplotlib.pyplot as plt
from curvature import *
from skimage import measure
import skimage.io

def get_curve(path, normalize=False, pad=True):
    """
    Extract a contour of a curve from an image using marching squares

    Parameters
    ----------
    path: string
        Path to image
    normalize: boolean
        Whether to perform RMS normalization on the curve
    
    Returns
    -------
    ndarray(N, 2)
        Sampled curve
    """

    ## Center the point cloud on its centroid and normalize
    #by its root mean square distance to the origin.  Note that this
    #does not change the normals at all, only the points, since it's a
    #uniform scale
    J = skimage.io.imread(path)
    I = J
    if pad:
        M, N = J.shape
        I = np.zeros((M*2, N*2))
        I[int(M/2):int(M/2)+M, int(N/2):int(N/2)+N] = J
    X = measure.find_contours(I, 0.5)[0]
    if normalize:
        X = X - np.mean(X, axis=0, keepdims=True)
        X = X*np.sqrt(X.shape[0]/np.sum(X**2))
    return X

def get_d2(X, n_samples, bins):
    N = X.shape[0]
    t = np.linspace(0, 1, N)
    s1 = np.random.rand(n_samples)
    Y1 = np.array([np.interp(s1, t, X[:, 0]), np.interp(s1, t, X[:, 1])]).T
    s2 = np.random.rand(n_samples)
    Y2 = np.array([np.interp(s2, t, X[:, 1]), np.interp(s2, t, X[:, 1])]).T
    d = np.sqrt(np.sum((Y1-Y2)**2, axis=1))
    return np.histogram(d, bins)[0]


if __name__ == '__main__2':
    n_samples = 1000000
    n_bins = 50
    bins = np.linspace(0, 3, n_bins+1)
    H = np.zeros((100, n_bins))
    for i in range(1, 101):
        print(".", end='')
        X = get_curve("blobs/resizedblob{}.png".format(i))
        H[i-1, :] = get_d2(X, n_samples, bins)

if __name__ == '__main__':
    X = get_curve("mpeg7/beetle-1.gif")
    X = X[0::4, :]
    sigmas = np.linspace(1, 150, 400)
    S = get_scale_space_images(X, 2, sigmas, loop=True)[1]
    N = 800
    scalespacefn = lambda X: get_scale_space_images(X, 2, sigmas, loop=True, n_arclen=N)[1]
    SEDT = get_scale_space_edt(X, scalespacefn)
    plt.imshow(SEDT, cmap='magma')
    plt.gca().invert_yaxis()
    plt.show()
