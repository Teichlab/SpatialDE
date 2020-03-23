""" Main underlying functions for SpatialDE functionality.
"""
import sys
import logging
from time import time
import warnings
from typing import Optional

import numpy as np
from scipy import stats
from scipy.special import logsumexp
import pandas as pd

from tqdm.auto import tqdm

from .kernels import SquaredExponential, Cosine, Linear
from .models import Model, Constant, Null, model_factory
from .util import bh_adjust


def get_l_limits(X):
    Xsq = np.sum(np.square(X), 1)
    R2 = -2.0 * np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
    R2 = np.clip(R2, 0, np.inf)
    R_vals = np.unique(R2.flatten())
    R_vals = R_vals[R_vals > 1e-8]

    l_min = np.sqrt(R_vals.min()) / 2.0
    l_max = np.sqrt(R_vals.max()) * 2.0

    return l_min, l_max


def simulate_const_model(MLL_params, N):
    dfm = np.zeros((N, MLL_params.shape[0]))
    for i, params in enumerate(MLL_params.iterrows()):
        params = params[1]
        s2_e = params.max_s2_t_hat * params.max_delta
        dfm[:, i] = np.random.normal(params.max_mu_hat, s2_e, N)

    dfm = pd.DataFrame(dfm)
    return dfm


def get_mll_results(results, null_model="const"):
    null_lls = results.query('model == "{}"'.format(null_model))[["g", "max_ll"]]
    model_results = results.query('model != "{}"'.format(null_model))
    model_results = model_results[
        model_results.groupby(["g", "model"])["max_ll"].transform(max)
        == model_results["max_ll"]
    ]
    mll_results = model_results.merge(null_lls, on="g", suffixes=("", "_null"))
    mll_results["LLR"] = mll_results["max_ll"] - mll_results["max_ll_null"]

    return mll_results


def inducers_grid(X, ninducers):
    rngmin = X.min(0)
    rngmax = X.max(0)
    xvals = np.linspace(rngmin[0], rngmax[0], int(np.ceil(np.sqrt(ninducers))))
    yvals = np.linspace(rngmin[1], rngmax[1], int(np.ceil(np.sqrt(ninducers))))
    xx, xy = np.meshgrid(xvals, yvals)
    return np.hstack((xx.reshape((xx.size, 1)), xy.reshape((xy.size, 1))))


def fit_model(
    model: Model,
    exp_tab: pd.DataFrame,
    null_predictions: Optional[dict] = None,
    null_variances: Optional[dict] = None,
):
    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i, gene in enumerate(tqdm(exp_tab.columns)):
            y = exp_tab.iloc[:, i].to_numpy()
            model.y = y
            t0 = time()

            res = model.optimize()
            t = time() - t0
            res = {
                "g": gene,
                "max_ll": model.log_marginal_likelihood,
                "max_delta": model.delta,
                "max_mu_hat": model.mu,
                "max_s2_t_hat": model.sigma_s2,
                "max_s2_e_hat": model.sigma_n2,
                "time": t,
                "n": model.n,
                "FSV": model.FSV,
                "s2_FSV": model.s2_FSV,
                "s2_logdelta": model.s2_logdelta,
                "converged": res.success,
                "M": model.n_parameters,
            }
            for (k, v) in vars(model.kernel).items():
                if k not in res:
                    res[k] = v
            if (
                null_predictions is not None
                and null_variances is not None
                and gene in null_predictions
                and gene in null_variances
            ):
                stest = model.score_test(null_predictions[gene], null_variances[gene])
                res["pval"] = stest.pval
                res["kappa"] = stest.kappa
                res["U_tilde"] = stest.U_tilde
                res["nu"] = stest.nu
            results.append(res)
    return pd.DataFrame(results)


def factory(kern: str, X: np.ndarray, lengthscale: Optional[float] = None):
    Z = None
    if X.shape[0] > 1000:
        Z = inducers_grid(X, np.maximum(100, np.sqrt(X.shape[0])))

    if kern == "linear":
        return model_factory(X, Z, Linear())
    elif kern == "SE":
        return model_factory(X, Z, SquaredExponential(lengthscale))
    elif kern == "PER":
        return model_factory(X, Z, Cosine(lengthscale))
    elif kern == "const":
        return Constant(X)
    elif kern == "null":
        return Null(X)
    else:
        raise ValueError("unknown kernel")


def kspace_walk(kernel_space: dict, X: np.ndarray):
    for kern, lengthscales in kernel_space.items():
        try:
            for l in lengthscales:
                yield factory(kern, X, l), kern
        except TypeError:
            yield factory(kern, X, lengthscales), kern

def dyn_de(
    X: pd.DataFrame, exp_tab: pd.DataFrame, kernel_space=None, null_model="const"
):
    if kernel_space == None:
        kernel_space = {"SE": [5.0, 25.0, 50.0]}

    results = []

    logging.info("Fitting null model")
    nullmodel = factory(null_model, X.to_numpy())
    null_fits = fit_model(nullmodel, exp_tab).dropna(axis=1)
    null_predictions = null_fits.set_index("g")["max_mu_hat"].to_dict()
    null_variances = null_fits.set_index("g")["max_s2_e_hat"].to_dict()

    logging.info("Fitting gene models")
    n_models = 0
    for model, mname in kspace_walk(kernel_space, X.to_numpy()):
        res = fit_model(model, exp_tab, null_predictions, null_variances)
        res["model"] = mname
        results.append(res)
        n_models += 1

    n_genes = exp_tab.shape[1]
    logging.info("Finished fitting {} models to {} genes".format(n_models, n_genes))

    results = pd.concat(results, sort=True).reset_index(drop=True)
    sizes = results.groupby(["model", "g"]).size().groupby("model").unique()
    results = results.set_index("model")
    results.loc[sizes > 1, "M"] += 1
    results = results.reset_index()
    results["BIC"] = -2 * results["max_ll"] + results["M"] * np.log(results["n"])

    return results, null_fits


def run(X, exp_tab, kernel_space=None, null_model="const"):
    """ Perform SpatialDE test

    X : matrix of spatial coordinates times observations
    exp_tab : Expression table, assumed appropriatealy normalised.

    The grid of covariance matrices to search over for the alternative
    model can be specifiec using the kernel_space parameter.
    """
    if kernel_space == None:
        l_min, l_max = get_l_limits(X)
        kernel_space = {
            "SE": np.logspace(np.log10(l_min), np.log10(l_max), 10),
            #'PER': np.logspace(np.log10(l_min), np.log10(l_max), 10),
            #'linear': None
        }

    logging.info("Performing DE test")
    results, null_fits = dyn_de(X, exp_tab, kernel_space, null_model)

    results = results.loc[results.groupby(["model", "g"])["max_ll"].idxmin()]
    results = results.loc[results.groupby("g")["BIC"].idxmin()]
    results["p.adj"] = bh_adjust(results["pval"].to_numpy())

    return results.merge(null_fits, on="g", suffixes=("", "_null"))

def model_search(X, exp_tab, DE_mll_results, kernel_space=None):
    """ Compare model fits with different models.

    This way DE genes can be classified to interpretable function classes.

    The strategy is based on ABCD in the Automatic Statistician, but
    using precomputed covariance matrices for all models in the search space.

    By default searches a grid of periodic covariance matrices and a linear
    covariance matrix.
    """
    if kernel_space == None:
        P_min, P_max = get_l_limits(X)
        kernel_space = {
            "PER": np.logspace(np.log10(P_min), np.log10(P_max), 10),
            "linear": 0,
        }

    de_exp_tab = exp_tab[DE_mll_results["g"]]

    logging.info("Performing model search")
    results = dyn_de(X, de_exp_tab, kernel_space)
    new_and_old_results = pd.concat((results, DE_mll_results), sort=True)

    # Calculate model probabilities
    mask = (
        new_and_old_results.groupby(["g", "model"])["BIC"].transform(min)
        == new_and_old_results["BIC"]
    )
    log_p_data_Hi = -new_and_old_results[mask].pivot_table(
        values="BIC", index="g", columns="model"
    )
    log_Z = logsumexp(log_p_data_Hi, 1)
    log_p_Hi_data = (log_p_data_Hi.T - log_Z).T
    p_Hi_data = np.exp(log_p_Hi_data).add_suffix("_prob")

    # Select most likely model
    mask = (
        new_and_old_results.groupby("g")["BIC"].transform(min)
        == new_and_old_results["BIC"]
    )
    ms_results = new_and_old_results[mask]

    ms_results = ms_results.join(p_Hi_data, on="g")

    # Retain information from significance testing in the new table
    transfer_columns = ["pval", "qval", "max_ll_null"]
    ms_results = ms_results.drop(transfer_columns, 1).merge(
        DE_mll_results[transfer_columns + ["g"]], on="g"
    )

    return ms_results
