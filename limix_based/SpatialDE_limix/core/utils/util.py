import numpy as np
import scipy.spatial as SS
import scipy as sp
from scipy import interpolate
from scipy import stats

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

def pval(alt, null):
    LLR = null.LML - alt.LML
    df = alt.N_params - null.N_params
    import pdb; pdb.set_trace()
    return 1. - stats.chi2.cdf(LLR, df=df)


def qvalue(pv, pi0=None):
    '''
    Estimates q-values from p-values

    This function is modified based on https://github.com/nfusi/qvalue

    Args
    ====
    pi0: if None, it's estimated as suggested in Storey and Tibshirani, 2003.

    '''
    assert(pv.min() >= 0 and pv.max() <= 1), "p-values should be between 0 and 1"

    original_shape = pv.shape
    pv = pv.ravel()  # flattens the array in place, more efficient than flatten()

    m = float(len(pv))

    # if the number of hypotheses is small, just set pi0 to 1
    if len(pv) < 100 and pi0 is None:
        pi0 = 1.0
    elif pi0 is not None:
        pi0 = pi0
    else:
        # evaluate pi0 for different lambdas
        pi0 = []
        lam = sp.arange(0, 0.90, 0.01)
        counts = sp.array([(pv > i).sum() for i in sp.arange(0, 0.9, 0.01)])
        for l in range(len(lam)):
            pi0.append(counts[l]/(m*(1-lam[l])))

        pi0 = sp.array(pi0)

        # fit natural cubic spline
        tck = interpolate.splrep(lam, pi0, k=3)
        pi0 = interpolate.splev(lam[-1], tck)

        if pi0 > 1:
            pi0 = 1.0

    assert(pi0 >= 0 and pi0 <= 1), "pi0 is not between 0 and 1: %f" % pi0

    p_ordered = sp.argsort(pv)
    pv = pv[p_ordered]
    qv = pi0 * m/len(pv) * pv
    qv[-1] = min(qv[-1], 1.0)

    for i in range(len(pv)-2, -1, -1):
        qv[i] = min(pi0*m*pv[i]/(i+1.0), qv[i+1])

    # reorder qvalues
    qv_temp = qv.copy()
    qv = sp.zeros_like(qv)
    qv[p_ordered] = qv_temp

    # reshape qvalues
    qv = qv.reshape(original_shape)

    return qv
