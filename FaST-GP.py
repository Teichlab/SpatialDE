import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import pandas as pd


def SE_kernel(X, l):
    Xsq = np.sum(np.square(X), 1)
    R2 = -2. * np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
    R2 = np.clip(R2, 0, np.inf)
    return np.exp(-R2 / (2 * l ** 2))


def factor(K):
    S, U = np.linalg.eigh(K)
    # .clip removes negative eigenvalues
    return U, S.clip(0.)


def get_UT1(U):
    return U.sum(0)


def get_UTy(U, y):
    return y.dot(U)


def mu_hat(delta, UTy, UT1, S, n):
    ''' ML Estimate of bias mu, function of delta.
    '''
    UT1_scaled = UT1 / (S + delta)
    sum_1 = UT1_scaled.dot(UTy)
    sum_2 = UT1_scaled.dot(UT1)

    return sum_1 / sum_2


def s2_t_hat(delta, UTy, S, n):
    ''' ML Estimate of structured noise, function of delta
    '''
    UTy_scaled = UTy / (S + delta)
    return UTy_scaled.dot(UTy) / n


def LL(delta, UTy, UT1, S, n):
    ''' Log-likelihood of GP model as a function of delta.

    The parameter delta is the ratio s2_e / s2_t, where s2_e is the
    observation noise and s2_t is the noise explained by covariance
    in time or space.
    '''
    mu_h = mu_hat(delta, UTy, UT1, S, n)

    sum_1 = (np.square(UTy - UT1 * mu_h) / (S + delta)).sum()
    sum_2 = np.log(S + delta).sum()

    return -0.5 * (n * np.log(2 * np.pi) + n * np.log(sum_1 / n) + sum_2 + n)


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

    max_mu_hat = mu_hat(max_delta, UTy, UT1, S, n)
    max_s2_t_hat = s2_t_hat(max_delta, UTy, S, n)

    return max_ll, max_delta, max_mu_hat, max_s2_t_hat


def lengthscale_fits(exp_tab, U, UT1, S, num=64):
    ''' Fit GPs after pre-processing for particular lengthscale
    '''
    results = []
    n, G = exp_tab.shape
    for g in tqdm(range(G)):
        y = exp_tab.iloc[:, g]
        UTy = get_UTy(U, y)

        max_ll, max_delta, max_mu_hat, max_s2_t_hat = search_max_LL(UTy, UT1, S, n, num)
        results.append({'g': exp_tab.columns[g],
                        'max_ll': max_ll,
                        'max_delta': max_delta,
                        'max_mu_hat': max_mu_hat,
                        'max_s2_t_hat': max_s2_t_hat})
        
    return pd.DataFrame(results)


def dyn_de(X, exp_tab, lengthscale=10, num=64):
    K = SE_kernel(X, lengthscale)
    U, S = factor(K)
    UT1 = get_UT1(U)
    results = lengthscale_fits(exp_tab, U, UT1, S, num)

    return results
