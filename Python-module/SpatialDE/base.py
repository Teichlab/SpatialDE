''' Main underlying functions for SpatialDE functionality.
'''
import sys
import logging
from time import time
import warnings

import numpy as np
import tensorflow as tf
import gpflow
from scipy import optimize
from scipy import linalg
from scipy import stats
from scipy.misc import derivative
from scipy.special import logsumexp

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from tqdm.autonotebook import tqdm

import pandas as pd

from .util import qvalue, Kernel, GP, SGPIPM, GPControl
from .gpflow_helpers import *

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

def factor(K):
    S, U = np.linalg.eigh(K)
    # .clip removes negative eigenvalues
    return U, np.clip(S, 1e-8, None)


def get_UT1(U):
    return U.sum(0)


def get_UTy(U, y):
    return y.dot(U)


def mu_hat(delta, UTy, UT1, S, n, Yvar=None):
    ''' ML Estimate of bias mu, function of delta.
    '''
    if Yvar is None:
        Yvar = np.ones_like(S)

    UT1_scaled = UT1 / (S + delta * Yvar)
    sum_1 = UT1_scaled.dot(UTy)
    sum_2 = UT1_scaled.dot(UT1)

    return sum_1 / sum_2


def s2_t_hat(delta, UTy, UT1, S, n, mu_hat, Yvar=None):
    ''' ML Estimate of structured noise, function of delta
    '''
    if Yvar is None:
        Yvar = np.ones_like(S)

    numerator = UTy - UT1 * mu_hat
    numerator_scaled = numerator / (S + delta * Yvar)
    return numerator_scaled.dot(numerator) / n


def LL(delta, UTy, UT1, S, n, Yvar=None):
    ''' Log-likelihood of GP model as a function of delta.

    The parameter delta is the ratio s2_e / s2_t, where s2_e is the
    observation noise and s2_t is the noise explained by covariance
    in time or space.
    '''

    mu_h = mu_hat(delta, UTy, UT1, S, n, Yvar)
    
    if Yvar is None:
        Yvar = np.ones_like(S)

    sum_1 = (np.square(UTy - UT1 * mu_h) / (S + delta * Yvar)).sum()
    sum_2 = np.log(S + delta * Yvar).sum()

    with np.errstate(divide='ignore'):
        return -0.5 * (n * np.log(2 * np.pi) + n * np.log(sum_1 / n) + sum_2 + n)

def make_objective(UTy, UT1, S, n, Yvar=None):
    def LL_obj(log_delta):
        return -LL(np.exp(log_delta), UTy, UT1, S, n, Yvar)

    return LL_obj

def lbfgsb_max_LL(UTy, UT1, S, n, Yvar=None):
    LL_obj = make_objective(UTy, UT1, S, n, Yvar)
    min_boundary = -10
    max_boundary = 20.
    x, f, d = optimize.fmin_l_bfgs_b(LL_obj, 0., approx_grad=True,
                                                 bounds=[(min_boundary, max_boundary)],
                                                 maxfun=64, factr=1e12, epsilon=1e-4)
    max_ll = -f
    max_delta = np.exp(x[0])

    boundary_ll = -LL_obj(max_boundary)
    if boundary_ll > max_ll:
        max_ll = boundary_ll
        max_delta = np.exp(max_boundary)

    boundary_ll = -LL_obj(min_boundary)
    if boundary_ll > max_ll:
        max_ll = boundary_ll
        max_delta = np.exp(min_boundary)


    max_mu_hat = mu_hat(max_delta, UTy, UT1, S, n, Yvar)
    max_s2_t_hat = s2_t_hat(max_delta, UTy, UT1, S, n, max_mu_hat, Yvar)

    s2_logdelta = 1. / (derivative(LL_obj, np.log(max_delta), n=2) ** 2)

    return max_ll, max_delta, max_mu_hat, max_s2_t_hat, s2_logdelta

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
            'max_mu_hat': max_mu_hat,
            'max_s2_s_hat': 0.,
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
            'max_mu_hat': max_mu_hat,
            'max_s2_s_hat': 0.,
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


def dyn_de(X, exp_tab, kernel_space=None, control=GPControl(), rng=np.random.default_rng()):
    if kernel_space == None:
        kernel_space = [Kernel.SE]

    results = []
    logging.info('Fitting gene models')
    for k in tqdm(kernel_space):
        if k is Kernel.null:
            result = null_fits(exp_tab)
            result['l'] = np.nan
            result['M'] = 1
            result['model'] = 'null'
            result['marginal_ll'] = result['max_ll'] - 0.5 * result['M'] * np.log(result['n'])
        elif k is Kernel.const:
            result = const_fits(exp_tab)
            result['l'] = np.nan
            result['M'] = 2
            result['model'] = 'const'
            result['marginal_ll'] = result['max_ll'] - 0.5 * result['M'] * np.log(result['n'])
        else:
            if k is Kernel.linear:
                kern = gpflow.kernels.Linear()
            elif k is Kernel.SE:
                kern = gpflow.kernels.SquaredExponential()
            elif k is Kernel.PER:
                kern = gpflow.kernels.Cosine()

            if control.gp == GP.GPR:
                model = GPRModel(X, exp_tab, kern)
            elif control.gp == GP.SPGPR:
                model = SGPRModel(X, exp_tab, kern, rng=rng, ipm=control.ipm, ninducers=control.ninducers)
            result = model.optimize()
            result.loc[:, 'model'] = k.name
        results.append(result)

    n_genes = exp_tab.shape[1]
    logging.info('Finished fitting {} models to {} genes'.format(len(kernel_space), n_genes))

    results = pd.concat(results, sort=True).reset_index(drop=True)

    return results


def run(X, exp_tab, kernel_space=None):
    ''' Perform SpatialDE test

    X : matrix of spatial coordinates times observations
    exp_tab : Expression table, assumed appropriatealy normalised.

    The grid of covariance matrices to search over for the alternative
    model can be specifiec using the kernel_space paramter.
    '''
    if kernel_space == None:
        kernel_space = [Kernel.SE, Kernel.const]

    logging.info('Performing DE test')
    results = dyn_de(X, exp_tab, kernel_space)
    mll_results = get_mll_results(results)

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
    if kernel_space is None:
        kernel_space = [Kernel.PER, Kernel.linear]

    de_exp_tab = exp_tab[DE_mll_results['g']]

    logging.info('Performing model search')
    results = dyn_de(X, de_exp_tab, kernel_space)
    new_and_old_results = pd.concat((results, DE_mll_results), sort=True)

    # Calculate model probabilities
    mask = new_and_old_results.groupby(['g', 'model'])['marginal_ll'].transform(max) == new_and_old_results['marginal_ll']
    log_p_data_Hi = -new_and_old_results[mask].pivot_table(values='marginal_ll', index='g', columns='model')
    log_Z = logsumexp(log_p_data_Hi, 1)
    log_p_Hi_data = (log_p_data_Hi.T - log_Z).T
    p_Hi_data = np.exp(log_p_Hi_data).add_suffix('_prob')

    # Select most likely model
    mask = new_and_old_results.groupby('g')['marginal_ll'].transform(max) == new_and_old_results['marginal_ll']
    ms_results = new_and_old_results[mask]

    ms_results = ms_results.join(p_Hi_data, on='g')

    # Retain information from significance testing in the new table
    transfer_columns = ['pval', 'qval', 'max_ll_null']
    ms_results = ms_results.drop(transfer_columns, 1) \
        .merge(DE_mll_results[transfer_columns + ['g']], on='g')

    return ms_results
