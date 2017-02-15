from time import time
import logging

import numpy as np
from scipy import optimize
from scipy import linalg
from scipy import stats
from tqdm import tqdm
import pandas as pd

logging.basicConfig(level=logging.DEBUG)


def SE_kernel(X, l):
    Xsq = np.sum(np.square(X), 1)
    R2 = -2. * np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
    R2 = np.clip(R2, 0, np.inf)
    return np.exp(-R2 / (2 * l ** 2))


def linear_kernel(X):
    return np.dot(X, X.T)


def cosine_kernel(X, p):
    ''' Periodic kernel as l -> oo in [Lloyd et al 2014]

    Easier interpretable composability with SE?
    '''
    Xsq = np.sum(np.square(X), 1)
    R2 = -2. * np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
    R2 = np.clip(R2, 0, np.inf)
    return np.cos(2 * np.pi * np.sqrt(R2) / p)


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
    # o = optimize.minimize_scalar(LL_obj)
    #  o.nfev has the number of function evals
    max_ll = -o.fun
    max_delta = np.exp(o.x)
    max_mu_hat = mu_hat(max_delta, UTy, UT1, S, n)
    max_s2_t_hat = s2_t_hat(max_delta, UTy, S, n)

    return max_ll, max_delta, max_mu_hat, max_s2_t_hat


def lbfgsb_max_LL(UTy, UT1, S, n):
    LL_obj = make_objective(UTy, UT1, S, n)
    x, f, d = optimize.fmin_l_bfgs_b(LL_obj, 0., approx_grad=True, bounds=[(-10, 10)],
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
    for g in tqdm(range(G)):
        y = exp_tab.iloc[:, g]
        UTy = get_UTy(U, y)

        t0 = time()
        max_ll, max_delta, max_mu_hat, max_s2_t_hat = lbfgsb_max_LL(UTy, UT1, S, n)
        # max_ll, max_delta, max_mu_hat, max_s2_t_hat = brent_max_LL(UTy, UT1, S, n)
        # max_ll, max_delta, max_mu_hat, max_s2_t_hat = search_max_LL(UTy, UT1, S, n, num)
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
    mll_results = model_results.merge(null_lls, on='g',)
    mll_results['D'] = mll_results['max_ll_x'] - mll_results['max_ll_y']

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
        UT1 = get_UT1(U)
        US_mats.append({
            'model': 'linear',
            'M': 3,
            'l': np.nan,
            'U': U,
            'S': S,
            'UT1': UT1
        })

    if 'SE' in kernel_space:
        for lengthscale in kernel_space['SE']:
            K = SE_kernel(X, lengthscale)
            U, S = factor(K)
            UT1 = get_UT1(U)
            US_mats.append({
                'model': 'SE',
                'M': 4,
                'l': lengthscale,
                'U': U,
                'S': S,
                'UT1': UT1
            })

    if 'PER' in kernel_space:
        for period in kernel_space['PER']:
            K = cosine_kernel(X, period)
            U, S = factor(K)
            UT1 = get_UT1(U)
            US_mats.append({
                'model': 'PER',
                'M': 4,
                'l': period,
                'U': U,
                'S': S,
                'UT1': UT1
            })

    t = time() - t0
    logging.info('Done: {0:.2}s'.format(t))

    logging.info('Fitting gene models')
    for cov in US_mats:
        result = lengthscale_fits(exp_tab, cov['U'], cov['UT1'], cov['S'])
        result['l'] = cov['l']
        result['M'] = cov['M']
        result['model'] = cov['model']
        results.append(result)

    results = pd.concat(results).reset_index(drop=True)
    results['BIC'] = -2 * results['max_ll'] + results['M'] * np.log(results['n'])

    return results


def run(X, exp_tab, kernel_space=None, pvalues=True, null_model_samples=10000):
    logging.info('Performing DE test')
    results = dyn_de(X, exp_tab, kernel_space)
    mll_results = get_mll_results(results)
    if not pvalues:
        return mll_results

    logging.info('Simulating {} null models'.format(null_model_samples))
    t0 = time()
    sim_null_exp_tab = simulate_const_model(mll_results.sample(null_model_samples), exp_tab.shape[0])
    t = time() - t0
    logging.info('Done: {0:.2}s'.format(t))

    logging.info('Performing DE test on null models'.format(null_model_samples))
    t0 = time()
    sim_results = dyn_de(X, sim_null_exp_tab, kernel_space)
    sim_mll_results = get_mll_results(sim_results)
    t = time() - t0
    logging.info('Done: {0:.2}s'.format(t))

    logging.info('Fitting null distribution')
    t0 = time()
    gamma_parameters = stats.gamma.fit(sim_mll_results['D'])
    gamma_rv = stats.gamma(*gamma_parameters)
    t = time() - t0
    logging.info('Done: {0:.2}s'.format(t))

    mll_results['pval'] = gamma_rv.sf(mll_results['D'])
    mll_results['qval'] = mll_results['pval'] * mll_results.shape[0]
    mll_results['qval'] = mll_results['qval'].map(lambda p: p if p < 1. else 1)

    return mll_results
