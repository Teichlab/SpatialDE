import numpy as np
import scipy.spatial as SS
import scipy as sp

def get_l_limits(X):
    Xsq = np.sum(np.square(X), 1)
    R2 = -2. * np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
    R2 = np.clip(R2, 0, np.inf)
    R_vals = np.unique(R2.flatten())
    R_vals = R_vals[R_vals > 1e-8]

    l_min = np.sqrt(R_vals.min()) / 2.
    l_max = np.sqrt(R_vals.max()) * 2.

    return l_min, l_max

def get_l_grid(X, n_grid = 10):
    l_min, l_max = get_l_limits(X)
    return np.logspace(np.log10(l_min), np.log10(l_max), n_grid)

def factor(K):
    S, U = np.linalg.eigh(K)
    # .clip removes negative eigenvalues
    return U, S.clip(0.)

def SE_kernel(X, l):
    rv = SS.distance.pdist(X,'euclidean')**2.
    rv = SS.distance.squareform(rv)
    return sp.exp(-rv/(2*l**2.))

    # Xsq = np.sum(np.square(X), 1)
    # R2 = -2. * np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
    # R2 = np.clip(R2, 0, np.inf)
    # return np.exp(-R2 / (2 * l ** 2))
