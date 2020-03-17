""" Main underlying functions for SpatialDE functionality.
"""
import sys
import logging
import numpy as np
import tensorflow as tf
import gpflow

from tqdm.auto import tqdm

import pandas as pd

from .util import bh_adjust, Kernel, GP, SGPIPM, GPControl
from .gpflow_helpers import *


def dyn_de(
    X, exp_tab, control: Optional[GPControl] = GPControl(), rng=np.random.default_rng()
):
    if control.gp is None:
        if X.shape[0] < 750:
            control.gp = GP.GPR
        else:
            control.gp = GP.SGPR

    results = DataSetResults()
    X = tf.constant(X.to_numpy())
    colnames = exp_tab.columns.to_numpy()
    t = tqdm(colnames)
    opt = gpflow.optimizers.Scipy()

    logging.info("Fitting gene models")
    if control.gp == GP.GPR:
        for g, gene in enumerate(t):
            t.set_description(gene, refresh=False)
            model = GPR(
                X,
                tf.constant(exp_tab.iloc[:, g].to_numpy()[:, np.newaxis]),
                control.ncomponents,
                control.ard,
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
            inducers.Z.trainable = False

        method = "BFGS"
        if control.ipm == SGPIPM.free and ninducers > 1e3:
            method = "L-BFGS-B"

        for g, gene in enumerate(t):
            t.set_description(gene, refresh=False)
            model = SGPR(
                X,
                tf.constant(exp_tab.iloc[:, g].to_numpy()[:, np.newaxis]),
                inducing_variable=inducers,
                n_kernel_components=control.ncomponents,
                ard=control.ard,
            )
            results[gene] = GeneGP(model, opt.minimize, method=method)

    logging.info("Finished fitting models to %i genes" % len(colnames))
    return results


def run(X, exp_tab):
    """ Perform SpatialDE test

    X : matrix of spatial coordinates times observations
    exp_tab : Expression table, assumed appropriatealy normalised.
    """
    logging.info("Performing DE test")
    results = dyn_de(X, exp_tab)

    df = results.to_df()
    df["p.adj"] = bh_adjust(df["pval"].to_numpy())

    return df, results


def model_search(X, exp_tab, DE_mll_results, kernel_space=None):
    """ Compare model fits with different models.

    This way DE genes can be classified to interpretable function classes.

    The strategy is based on ABCD in the Automatic Statistician, but
    using precomputed covariance matrices for all models in the search space.

    By default searches a grid of periodic covariance matrices and a linear
    covariance matrix.
    """
    if kernel_space is None:
        kernel_space = [Kernel.PER, Kernel.linear]

    de_exp_tab = exp_tab[DE_mll_results["g"]]

    logging.info("Performing model search")
    results = dyn_de(X, de_exp_tab, kernel_space)
    new_and_old_results = pd.concat((results, DE_mll_results), sort=True)

    # Calculate model probabilities
    mask = (
        new_and_old_results.groupby(["g", "model"])["marginal_ll"].transform(max)
        == new_and_old_results["marginal_ll"]
    )
    log_p_data_Hi = -new_and_old_results[mask].pivot_table(
        values="marginal_ll", index="g", columns="model"
    )
    log_Z = logsumexp(log_p_data_Hi, 1)
    log_p_Hi_data = (log_p_data_Hi.T - log_Z).T
    p_Hi_data = np.exp(log_p_Hi_data).add_suffix("_prob")

    # Select most likely model
    mask = (
        new_and_old_results.groupby("g")["marginal_ll"].transform(max)
        == new_and_old_results["marginal_ll"]
    )
    ms_results = new_and_old_results[mask]

    ms_results = ms_results.join(p_Hi_data, on="g")

    # Retain information from significance testing in the new table
    transfer_columns = ["pval", "qval", "max_ll_null"]
    ms_results = ms_results.drop(transfer_columns, 1).merge(
        DE_mll_results[transfer_columns + ["g"]], on="g"
    )

    return ms_results
