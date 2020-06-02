""" Main underlying functions for SpatialDE functionality.
"""
import sys
import logging
from time import time
import warnings
from typing import Optional, Dict, Tuple

import numpy as np
from scipy import stats
from scipy.special import logsumexp
import pandas as pd

import tensorflow as tf
import tensorflow_probability as tfp
from .kernels import DistanceCache, SquaredExponential, Cosine, Linear

from tqdm.auto import tqdm

from .models import Model, Constant, Null, model_factory
from .util import bh_adjust, Kernel, GP, SGPIPM, GPControl
from .score_test import (
    NegativeBinomialScoreTest,
    combine_pvalues,
)
from .gpflow_helpers import *

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


def get_l_limits(cache: DistanceCache):
    R2 = cache.squaredEuclideanDistance
    R2 = R2[R2 > 1e-8]

    l_min = tf.sqrt(tf.reduce_min(R2)) * 2.0
    l_max = tf.sqrt(tf.reduce_max(R2))

    return l_min, l_max


def inducers_grid(X, ninducers):
    rngmin = X.min(0)
    rngmax = X.max(0)
    xvals = np.linspace(rngmin[0], rngmax[0], int(np.ceil(np.sqrt(ninducers))))
    yvals = np.linspace(rngmin[1], rngmax[1], int(np.ceil(np.sqrt(ninducers))))
    xx, xy = np.meshgrid(xvals, yvals)
    return np.hstack((xx.reshape((xx.size, 1)), xy.reshape((xy.size, 1))))


def fit_model(model: Model, exp_tab: pd.DataFrame, raw_counts: pd.DataFrame):
    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with model:
            for i, gene in enumerate(tqdm(exp_tab.columns)):
                y = exp_tab.iloc[:, i].to_numpy()
                rawy = raw_counts.iloc[:, i].to_numpy()
                model.sety(y, rawy)
                t0 = time()

                res = model.optimize()
                t = time() - t0
                res = {
                    "gene": gene,
                    "max_ll": model.log_marginal_likelihood,
                    "max_delta": model.delta,
                    "max_mu_hat": model.mu,
                    "max_s2_t_hat": model.sigma_s2,
                    "max_s2_e_hat": model.sigma_n2,
                    "time": t,
                    "n": model.n,
                    "FSV": model.FSV,
                    "s2_FSV": np.abs(
                        model.s2_FSV
                    ),  # we are at the optimum, so this should never be negative.
                    "s2_logdelta": np.abs(
                        model.s2_logdelta
                    ),  # Negative results are due to numerical errors when evaluating vanishing Hessians
                    "converged": res.success,
                    "M": model.n_parameters,
                }
                for (k, v) in vars(model.kernel).items():
                    if k not in res:
                        res[k] = v

                results.append(res)
    return pd.DataFrame(results)


def factory(kern: str, cache: DistanceCache, lengthscale: Optional[float] = None):
    if kern == "linear":
        return Linear(cache)
    elif kern == "SE":
        return SquaredExponential(cache, lengthscale=lengthscale)
    elif kern == "PER":
        return Cosine(cache, lengthscale=lengthscale)
    else:
        raise ValueError("unknown kernel")


def kspace_walk(kernel_space: dict, cache: DistanceCache):
    for kern, lengthscales in kernel_space.items():
        try:
            for l in lengthscales:
                yield factory(kern, cache, l), kern
        except TypeError:
            yield factory(kern, cache, lengthscales), kern


def default_kernel_space(X: np.ndarray, cache: DistanceCache):
    l_min, l_max = get_l_limits(cache)
    return {
        "SE": np.logspace(np.log10(l_min), np.log10(l_max), 10),
        # "PER": np.logspace(np.log10(l_min), np.log10(l_max), 20),
        #'linear': None
    }

def _add_individual_score_test_result(resultdict, kernel, kname, gene):
    if "kernel" not in resultdict:
        resultdict["kernel"] = [kname]
    else:
        resultdict["kernel"].append(kname)
    if "gene" not in resultdict:
        resultdict["gene"] = [gene]
    else:
        resultdict["gene"].append(gene)
    for key, var in vars(kernel).items():
        if key[0] != "_":
            if key not in resultdict:
                resultdict[key] = [var]
            else:
                resultdict[key].append(var)
    return resultdict


def dyn_de(
    X: pd.DataFrame,
    raw_counts: pd.DataFrame,
    omnibus: bool = False,
    kernel_space: Optional[dict] = None,
) -> pd.DataFrame:
    logging.info("Performing DE test")

    X = X.to_numpy()
    dcache = DistanceCache(X)
    if kernel_space is None:
        kernel_space = default_kernel_space(X, dcache)

    individual_results = None if omnibus else []
    if X.shape[0] <= 2000 or omnibus:
        kernels = []
        kernelnames = []
        for k, name in kspace_walk(kernel_space, dcache):
            kernels.append(k)
            kernelnames.append(name)
        test = NegativeBinomialScoreTest(
            X,
            None,
            raw_counts.to_numpy(),
            omnibus,
            kernels,
        )

        results = []
        for i, g in enumerate(tqdm(raw_counts.columns)):
            t0 = time()
            result = test(i)
            t = time() - t0
            res = {"gene": g, "time": t}
            resultdict = result.to_dict()
            if omnibus:
                res.update(resultdict)
            else:
                res["pval"] = combine_pvalues(result).numpy()
            results.append(res)
            if not omnibus:
                for k, n in zip(kernels, kernelnames):
                    _add_individual_score_test_result(resultdict, k, n, g)
                individual_results.append(resultdict)

    else:  # doing all tests at once with stacked kernels leads to excessive memory consumption
        results = [[0, []] for _ in range(raw_counts.shape[1])]
        test = NegativeBinomialScoreTest(X, None, raw_counts.to_numpy())
        for k, n in kspace_walk(kernel_space, dcache):
            test.kernel = k
            for i in tqdm(range(len(raw_counts.columns))):
                t0 = time()
                res = test(i)
                t = time() - t0
                results[i][0] += t
                results[i][1].append(res)
                resultdict = res.to_dict()
                individual_results.append(_add_individual_score_test_result(resultdict, k, n, g))
        for i, g in enumerate(raw_counts.columns):
            results[i] = {
                "gene": g,
                "time": results[i][0],
                "pval": combine_pvalues(results[i][1]).numpy(),
            }

    results = pd.DataFrame(results)
    results["p.adj"] = bh_adjust(results.pval.to_numpy())

    if individual_results is not None:
        merged = {}
        for res in individual_results:
            for k, v in res.items():
                if k not in merged:
                    merged[k] = v
                else:
                    if isinstance(merged[k], np.ndarray):
                        merged[k] = np.concatenate((merged[k], v))
                    else:
                        merged[k].extend(v)
        individual_results = pd.DataFrame(merged)
    return results, individual_results


def run_gpflow(
    X: pd.DataFrame,
    exp_tab: pd.DataFrame,
    raw_counts: Optional[pd.DataFrame] = None,
    control: Optional[GPControl] = GPControl(),
    rng: np.random.Generator = np.random.default_rng(),
):
    if control.gp is None:
        if X.shape[0] < 750:
            control.gp = GP.GPR
        else:
            control.gp = GP.SGPR

    results = DataSetResults()
    X = tf.constant(X.to_numpy(), dtype=gpflow.config.default_float())
    colnames = exp_tab.columns.to_numpy()
    t = tqdm(colnames)
    opt = gpflow.optimizers.Scipy()

    logging.info("Fitting gene models")
    if control.gp == GP.GPR:
        for g, gene in enumerate(t):
            t.set_description(gene, refresh=False)
            model = GPR(
                X,
                Y=tf.constant(
                    exp_tab.iloc[:, g].to_numpy()[:, np.newaxis],
                    dtype=gpflow.config.default_float(),
                ),
                rawY=tf.constant(raw_counts.iloc[:, g].to_numpy()[:, np.newaxis])
                if raw_counts is not None
                else None,
                n_kernel_components=control.ncomponents,
                ard=control.ard,
            )
            results[gene] = GeneGP(model, opt.minimize, method="bfgs")
    elif control.gp == GP.SGPR:
        ninducers = (
            np.ceil(np.sqrt(X.shape[0])).astype(np.int32)
            if control.ninducers is None
            else control.ninducers
        )
        if control.ipm == SGPIPM.free or control.ipm == SGPIPM.random:
            inducers = X.iloc[rng.integers(0, X.shape[0], ninducers), :].to_numpy()
        elif control.ipm == SGPIPM.grid:
            rngmin = tf.reduce_min(X, axis=0)
            rngmax = tf.reduce_max(X, axis=0)
            xvals = tf.linspace(rngmin[0], rngmax[0], int(np.ceil(np.sqrt(ninducers))))
            yvals = tf.linspace(rngmin[1], rngmax[1], int(np.ceil(np.sqrt(ninducers))))
            xx, xy = tf.meshgrid(xvals, yvals)
            inducers = tf.stack((tf.reshape(xx, (-1,)), tf.reshape(xy, (-1,))), axis=1)
            inducers = gpflow.inducing_variables.InducingPoints(inducers)
        if control.ipm != SGPIPM.free:
            gpflow.utilities.set_trainable(inducers, False)

        method = "BFGS"
        if control.ipm == SGPIPM.free and ninducers > 1e3:
            method = "L-BFGS-B"

        for g, gene in enumerate(t):
            t.set_description(gene, refresh=False)
            model = SGPR(
                X,
                Y=tf.constant(
                    exp_tab.iloc[:, g].to_numpy()[:, np.newaxis],
                    dtype=gpflow.config.default_float(),
                ),
                rawY=tf.constant(raw_counts.iloc[:, g].to_numpy()[:, np.newaxis])
                if raw_counts is not None
                else None,
                inducing_variable=inducers,
                n_kernel_components=control.ncomponents,
                ard=control.ard,
            )
            results[gene] = GeneGP(model, opt.minimize, method=method)

    logging.info("Finished fitting models to %i genes" % len(colnames))
    return results


def run_fast(
    X: pd.DataFrame,
    exp_tab: pd.DataFrame,
    raw_counts: pd.DataFrame,
    kernel_space: Optional[dict] = None,
) -> pd.DataFrame:
    """ Perform SpatialDE test

    X : matrix of spatial coordinates times observations
    exp_tab : Expression table, assumed appropriatealy normalised.
    raw_counts : Unnormalized expression table

    The grid of covariance matrices to search over for the alternative
    model can be specifiec using the kernel_space parameter.
    """
    X = X.to_numpy()
    dcache = DistanceCache(X)
    if kernel_space == None:
        l_min, l_max = get_l_limits(cache)
        kernel_space = {
            "SE": np.logspace(np.log10(l_min), np.log10(l_max), 10),
            #'PER': np.logspace(np.log10(l_min), np.log10(l_max), 10),
            #'linear': None
        }

    logging.info("Fitting gene models")
    n_models = 0
    Z = None
    if X.shape[0] > 1000:
        Z = inducers_grid(X, np.maximum(100, np.sqrt(X.shape[0])))

    results = []
    for kern, kname in kspace_walk(kernel_space, dcache):
        model = model_factory(X.to_numpy(), Z, kern)
        res = fit_model(model, exp_tab, raw_counts)
        res["model"] = kname
        results.append(res)
        n_models += 1

    n_genes = exp_tab.shape[1]
    logging.info("Finished fitting {} models to {} genes".format(n_models, n_genes))

    results = pd.concat(results, sort=True).reset_index(drop=True)
    sizes = (
        results.groupby(["model", "gene"], sort=False).size().groupby("model", sort=False).unique()
    )
    results = results.set_index("model")
    results.loc[sizes > 1, "M"] += 1
    results = results.reset_index()
    results["BIC"] = -2 * results["max_ll"] + results["M"] * np.log(results["n"])

    results = results.loc[results.groupby(["model", "gene"], sort=False)["max_ll"].idxmax()]
    results = results.loc[results.groupby("gene", sort=False)["BIC"].idxmin()]

    return results.reset_index(drop=True)


def run_detailed(
    X: pd.DataFrame,
    exp_tab: pd.DataFrame,
    raw_counts: pd.DataFrame,
    control: GPControl = GPControl(),
    rng: np.random.Generator = np.random.default_rng(),
):
    logging.info("Fitting gene models")
    res = run_gpflow(X, exp_tab, raw_counts, control, rng)
    logging.info("Finished fitting models to {} genes".format(X.shape[0]))
    results = res.to_df(modelcol="model")

    return results.reset_index(drop=True)


def run(
    X: pd.DataFrame,
    exp_tab: pd.DataFrame,
    raw_counts: pd.DataFrame,
    control: Optional[GPControl] = GPControl,
    kernel_space: Optional[dict] = None,
    rng: np.random.Generator = np.random.default_rng(),
):
    if control is None:
        return run_fast(X, exp_tab, raw_counts, kernel_space)
    else:
        return run_detailed(X, exp_tab, raw_counts, control, rng)
