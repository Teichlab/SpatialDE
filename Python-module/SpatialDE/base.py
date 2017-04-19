import sys
from time import time
import logging

import numpy as np
from scipy import optimize
from scipy import linalg
from scipy import stats
from tqdm import tqdm
import pandas as pd

from .util import qvalue

logging.basicConfig(level=logging.DEBUG)


def get_l_limits(X):
    Xsq = np.sum(np.square(X), 1)
    R2 = -2. * np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
    R2 = np.clip(R2, 0, np.inf)
    R_vals = np.unique(R2.flatten())
    R_vals = R_vals[R_vals > 1e-8]
    
    l_min = np.sqrt(R_vals.min()) / 2.
    l_max = np.sqrt(R_vals.max()) * 2.
    
    return l_min, l_max

## Kernels ##

def SE_kernel(X, l):
    Xsq = np.sum(np.square(X), 1)
    R2 = -2. * np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
    R2 = np.clip(R2, 1e-12, np.inf)
    return np.exp(-R2 / (2 * l ** 2))


def linear_kernel(X):
    K = np.dot(X, X.T)
    return K / K.max()


def cosine_kernel(X, p):
    ''' Periodic kernel as l -> oo in [Lloyd et al 2014]

    Easier interpretable composability with SE?
    '''
    Xsq = np.sum(np.square(X), 1)
    R2 = -2. * np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
    R2 = np.clip(R2, 1e-12, np.inf)
    return np.cos(2 * np.pi * np.sqrt(R2) / p)


def gower_scaling_factor(K):
    ''' Gower normalization factor for covariance matric K

    Based on https://github.com/PMBio/limix/blob/master/limix/utils/preprocess.py
    '''
    n = K.shape[0]
    P = np.eye(n) - np.ones((n, n)) / n
    KP = K - K.mean(0)[:, np.newaxis]
    trPKP = np.sum(P * KP)

    return trPKP / (n - 1)


def factor(K):
    S, U = linalg.eigh(K)
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

    with np.errstate(divide='ignore'):
        return -0.5 * (n * np.log(2 * np.pi) + n * np.log(sum_1 / n) + sum_2 + n)


def make_objective(UTy, UT1, S, n):
    def LL_obj(log_delta):
        return -LL(np.exp(log_delta), UTy, UT1, S, n)

    return LL_obj


def brent_max_LL(UTy, UT1, S, n):
    LL_obj = make_objective(UTy, UT1, S, n)
    o = optimize.minimize_scalar(LL_obj, bounds=[-10, 10], method='bounded', options={'maxiter': 32})
    max_ll = -o.fun
    max_delta = np.exp(o.x)
    max_mu_hat = mu_hat(max_delta, UTy, UT1, S, n)
    max_s2_t_hat = s2_t_hat(max_delta, UTy, S, n)

    return max_ll, max_delta, max_mu_hat, max_s2_t_hat


def lbfgsb_max_LL(UTy, UT1, S, n):
    LL_obj = make_objective(UTy, UT1, S, n)
    x, f, d = optimize.fmin_l_bfgs_b(LL_obj, 0., approx_grad=True, bounds=[(-10, 20)],
                                                 maxfun=32, factr=1e12, epsilon=1e-4)
    max_ll = -f
    max_delta = np.exp(x[0])
    max_mu_hat = mu_hat(max_delta, UTy, UT1, S, n)
    max_s2_t_hat = s2_t_hat(max_delta, UTy, S, n)

    return max_ll, max_delta, max_mu_hat, max_s2_t_hat


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
    for g in tqdm(range(G), leave=False):
        y = exp_tab.iloc[:, g]
        UTy = get_UTy(U, y)

        t0 = time()
        max_ll, max_delta, max_mu_hat, max_s2_t_hat = lbfgsb_max_LL(UTy, UT1, S, n)
        t = time() - t0
        
        results.append({
            'g': exp_tab.columns[g],
            'max_ll': max_ll,
            'max_delta': max_delta,
            'max_mu_hat': max_mu_hat,
            'max_s2_t_hat': max_s2_t_hat,
            'time': t,
            'n': n
        })
        
    return pd.DataFrame(results)


def null_fits(exp_tab):
    ''' Get maximum LL for null model
    '''
    results = []
    n, G = exp_tab.shape
    for g in range(G):
        y = exp_tab.iloc[:, g]
        max_mu_hat = 0.
        max_s2_e_hat = np.square(y).sum() / n  # mll estimate
        max_ll = -0.5 * (n * np.log(2 * np.pi) + n + n * np.log(max_s2_e_hat))

        results.append({
            'g': exp_tab.columns[g],
            'max_ll': max_ll,
            'max_delta': np.inf,
            'max_mu_hat': max_mu_hat,
            'max_s2_t_hat': 0.,
            'time': 0,
            'n': n
        })
    
    return pd.DataFrame(results)


def const_fits(exp_tab):
    ''' Get maximum LL for const model
    '''
    results = []
    n, G = exp_tab.shape
    for g in range(G):
        y = exp_tab.iloc[:, g]
        max_mu_hat = y.mean()
        max_s2_e_hat = y.var()
        sum1 = np.square(y - max_mu_hat).sum()
        max_ll = -0.5 * ( n * np.log(max_s2_e_hat) + sum1 / max_s2_e_hat + n * np.log(2 * np.pi) )

        results.append({
            'g': exp_tab.columns[g],
            'max_ll': max_ll,
            'max_delta': np.inf,
            'max_mu_hat': max_mu_hat,
            'max_s2_t_hat': 0.,
            'time': 0,
            'n': n
        })
    
    return pd.DataFrame(results)


def simulate_const_model(MLL_params, N):
    dfm = np.zeros((N, MLL_params.shape[0]))
    for i, params in enumerate(MLL_params.iterrows()):
        params = params[1]
        s2_e = params.max_s2_t_hat * params.max_delta
        dfm[:, i] = np.random.normal(params.max_mu_hat, s2_e, N)
        
    dfm = pd.DataFrame(dfm)
    return dfm


def get_mll_results(results, null_model='const'):
    null_lls = results.query('model == "{}"'.format(null_model))[['g', 'max_ll']]
    model_results = results.query('model != "{}"'.format(null_model))
    model_results = model_results[model_results.groupby(['g'])['max_ll'].transform(max) == model_results['max_ll']]
    mll_results = model_results.merge(null_lls, on='g', suffixes=('', '_null'))
    mll_results['LLR'] = mll_results['max_ll'] - mll_results['max_ll_null']

    return mll_results


def dyn_de(X, exp_tab, kernel_space=None):
    if kernel_space == None:
        kernel_space = {
            'SE': [5., 25., 50.]
        }

    results = []

    if 'null' in kernel_space:
        result = null_fits(exp_tab)
        result['l'] = np.nan
        result['M'] = 1
        result['model'] = 'null'
        results.append(result)

    if 'const' in kernel_space:
        result = const_fits(exp_tab)
        result['l'] = np.nan
        result['M'] = 2
        result['model'] = 'const'
        results.append(result)

    logging.info('Pre-calculating USU^T = K\'s ...')
    US_mats = []
    t0 = time()
    if 'linear' in kernel_space:
        K = linear_kernel(X)
        U, S = factor(K)
        gower = gower_scaling_factor(K)
        UT1 = get_UT1(U)
        US_mats.append({
            'model': 'linear',
            'M': 3,
            'l': np.nan,
            'U': U,
            'S': S,
            'UT1': UT1,
            'Gower': gower
        })

    if 'SE' in kernel_space:
        for lengthscale in kernel_space['SE']:
            K = SE_kernel(X, lengthscale)
            U, S = factor(K)
            gower = gower_scaling_factor(K)
            UT1 = get_UT1(U)
            US_mats.append({
                'model': 'SE',
                'M': 4,
                'l': lengthscale,
                'U': U,
                'S': S,
                'UT1': UT1,
                'Gower': gower
            })

    if 'PER' in kernel_space:
        for period in kernel_space['PER']:
            K = cosine_kernel(X, period)
            U, S = factor(K)
            gower = gower_scaling_factor(K)
            UT1 = get_UT1(U)
            US_mats.append({
                'model': 'PER',
                'M': 4,
                'l': period,
                'U': U,
                'S': S,
                'UT1': UT1,
                'Gower': gower
            })

    t = time() - t0
    logging.info('Done: {0:.2}s'.format(t))

    logging.info('Fitting gene models')
    n_models = len(US_mats)
    for i, cov in enumerate(US_mats):
        logging.info('Model {} of {}'.format(i + 1, n_models))
        result = lengthscale_fits(exp_tab, cov['U'], cov['UT1'], cov['S'])
        result['l'] = cov['l']
        result['M'] = cov['M']
        result['model'] = cov['model']
        result['Gower'] = cov['Gower']
        results.append(result)

    results = pd.concat(results).reset_index(drop=True)
    results['BIC'] = -2 * results['max_ll'] + results['M'] * np.log(results['n'])

    return results


def run(X, exp_tab, kernel_space=None):
    ''' Perform SpatialDE test

    X : matrix of spatial coordinates times observations
    exp_tab : Expression table, assumed appropriatealy normalised.

    The grid of covariance matrices to search over for the alternative
    model can be specifiec using the kernel_space paramter.
    '''
    if kernel_space == None:
        l_min, l_max = get_l_limits(X)
        kernel_space = {
            'SE': np.logspace(np.log10(l_min), np.log10(l_max), 10),
            'const': 0
        }

    logging.info('Performing DE test')
    results = dyn_de(X, exp_tab, kernel_space)
    mll_results = get_mll_results(results)

    # Quantify fraction spatial variance
    scaled_spatial_var = mll_results['max_s2_t_hat'] * mll_results['Gower']
    noise_var = mll_results['max_s2_t_hat'] * mll_results['max_delta']
    mll_results['fraction_spatial_variance'] = scaled_spatial_var / (scaled_spatial_var + noise_var)

    # Perform significance test
    mll_results['pval'] = 1 - stats.chi2.cdf(mll_results['LLR'], df=1)
    mll_results['qval'] = qvalue(mll_results['pval'])

    return mll_results


def model_search(X, exp_tab, DE_mll_results, kernel_space=None):
    ''' Compare model fits with different models.

    This way DE genes can be classified to interpretable function classes.

    The strategy is based on ABCD in the Automatic Statistician, but
    using precomputed covariance matrices for all models in the search space.

    By default searches a grid of periodic covariance matrices and a linear
    covariance matrix.
    '''
    if kernel_space == None:
        P_min, P_max = get_l_limits(X)
        kernel_space = {
            'PER': np.logspace(np.log10(P_min), np.log10(P_max), 10),
            'linear': 0
        }

    de_exp_tab = exp_tab[DE_mll_results['g']]

    logging.info('Performing model search')
    results = dyn_de(X, de_exp_tab, kernel_space)
    new_and_old_results = pd.concat((results, DE_mll_results))
    mask = new_and_old_results.groupby('g')['BIC'].transform(min) == new_and_old_results['BIC']
    ms_results = new_and_old_results[mask]

    # Retain information from significance testing in the new table
    transfer_columns = ['pval', 'qval', 'max_ll_null']
    ms_results = ms_results.drop(transfer_columns, 1) \
        .merge(DE_mll_results[transfer_columns + ['g']], on='g')

    return ms_results
