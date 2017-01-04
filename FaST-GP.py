import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import pandas as pd


def SE_kernel(X, l):
    R = squareform(pdist(X, 'euclidean')) ** 2
    return np.exp(R / (2 * l ** 2))


def factor(K):
    S, U = np.linalg.eigh(K)
    # .clip removes negative eigenvalues
    return U, S.clip(0.)


def get_UT1(U):
    return U.sum(1)


def get_UTy(U, y):
    return y.dot(U)


def mu_hat(delta, UTy, UT1, Sd, n):
    ''' ML Estimate of bias mu, function of delta.
    '''
    UT1_scaled = UT1 / Sd
    sum_1 = (UT1_scaled).dot(UT1)
    sum_2 = (UT1_scaled).dot(UTy)

    return sum_2 / sum_1


def LL(delta, UTy, UT1, S, n):
    ''' Log-likelihood of GP model as a function of delta.

    The parameter delta is the ratio s_e / s_t, where s_e is the
    observation noise and s_t is the noise explained by covariance
    in time or space.
    '''
    Sd = (S + delta)
    mu_h = mu_hat(delta, UTy, UT1, Sd, n)
    sum_1 = np.log(Sd).sum()
    sum_2 = ((UTy - UT1 * mu_h) / Sd).sum()

    return -0.5 * (n * np.log(2 * np.pi) + sum_1 + n + n * np.log(sum_2))


def search_max_LL(UTy, UT1, S, n, num=64):
    ''' Search for delta which maximizes log likelihood.
    '''
    max_ll = -np.inf
    max_delta = np.nan
    for delta in np.logspace(base=np.e, start=-10, stop=10, num=num):
        cur_ll = LL(delta, UTy, UT1, S, n)
        if cur_ll > max_ll:
            max_ll = cur_ll
            max_delta = delta

    return max_ll, max_delta


def lengthscale_fits(exp_tab, U, UT1, S, num=64):
    ''' Fit GPs after pre-processing for particular lengthscale
    '''
    results = []
    n, G = exp_tab.shape
    print(G)
    for g in tqdm(range(G)):
        y = exp_tab.iloc[:, g]
        UTy = get_UTy(U, y)

        max_ll, max_delta = search_max_LL(UTy, UT1, S, n, num)
        results.append({'g': exp_tab.columns[g],
                        'max_ll': max_ll,
                        'max_delta': max_delta})
        
    return pd.DataFrame(results)


def dyn_de(X, exp_tab, lengthscale=10, num=64):
    K = SE_kernel(X, lengthscale)
    U, S = factor(K)
    UT1 = get_UT1(U)
    results = lengthscale_fits(exp_tab, U, UT1, S, num)

    return results
