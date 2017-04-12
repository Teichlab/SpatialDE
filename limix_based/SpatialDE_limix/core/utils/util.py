import numpy as np 

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
