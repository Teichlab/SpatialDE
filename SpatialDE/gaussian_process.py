import logging
from time import time
import warnings
from typing import Optional, Dict, List
from enum import Enum, auto

import numpy as np
import pandas as pd

from tqdm.auto import tqdm

import tensorflow as tf

import NaiveDE
from anndata import AnnData

from ._internal.kernels import SquaredExponential, Cosine, Linear
from ._internal.models import Model, Constant, Null, model_factory
from ._internal.util import (
    DistanceCache,
    default_kernel_space,
    kspace_walk,
    dense_slice,
    normalize_counts,
)
from ._internal.tf_dataset import AnnDataDataset
from ._internal.gpflow_helpers import *


class GP(Enum):
    GPR = auto()
    """
    Dense Gaussian process.
    """
    SGPR = auto()
    """
    Sparse Gaussian process.
    """


class SGPIPM(Enum):
    free = auto()
    """Inducing points are initialized randomly and their positions are optimized together with the other parameters."""
    random = auto()
    """Inducing points are placed at random locations."""
    grid = auto()
    """Inducing points are placed in a regular grid."""


@dataclass(frozen=True)
class GPControl:
    """
    Parameters for Gaussian process fitting.

    Args:
        gp: Type of GP to fit.
        ipm: Inducing point method. Only used if ``gp == GP.SGPR``.
        ncomponents: Number of kernel components.
        ard: Whether to use automatic relevance determination. This amounts to having one
            lengthscale per spatial dimension.
        ninducers: Number of inducing points.
    """

    gp: Optional[GP] = None
    ipm: SGPIPM = SGPIPM.grid
    ncomponents: int = 5
    ard: bool = True
    ninducers: Optional[int] = None


def inducers_grid(X, ninducers):
    rngmin = X.min(0)
    rngmax = X.max(0)
    xvals = np.linspace(rngmin[0], rngmax[0], int(np.ceil(np.sqrt(ninducers))))
    yvals = np.linspace(rngmin[1], rngmax[1], int(np.ceil(np.sqrt(ninducers))))
    xx, xy = np.meshgrid(xvals, yvals)
    return np.hstack((xx.reshape((xx.size, 1)), xy.reshape((xy.size, 1))))


def fit_model(model: Model, genes: Union[List[str], np.ndarray], counts: np.ndarray):
    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with model:
            for i, gene in enumerate(tqdm(genes)):
                y = dense_slice(counts[:, i])
                model.y = y
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
                    if k not in res and k[0] != "_":
                        res[k] = v

                results.append(res)
    return pd.DataFrame(results)


def fit_detailed(
    adata: AnnData,
    genes: Optional[List[str]] = None,
    layer: Optional[str] = None,
    normalized: bool = False,
    sizefactor_col: Optional[str] = None,
    spatial_key: str = "spatial",
    control: Optional[GPControl] = GPControl(),
    rng: np.random.Generator = np.random.default_rng(),
) -> DataSetResults:
    """
    Fits Gaussian processes to genes.

    A Gaussian process based on highly interpretable spectral mixture kernels (Wilson et al. 2013, Wilson 2014) is fitted
    separately to each gene. Sparse GPs are used on large datasets (>1000 observations) to improve speed.
    This uses a Gaussian likelihood and requires appropriate data normalization.

    Args:
        adata: The annotated data matrix.
        genes: List of genes to base the analysis on. Defaults to all genes.
        layer: Name of the AnnData object layer to use. By default ``adata.X`` is used.
        normalized: Whether the data are already normalized to an approximately Gaussian likelihood.
            If ``False``, they will be normalized using the workflow from Svensson et al, 2018.
        sizefactor_col: Column in ``adata.obs`` to be used for normalization. If ``None``, total number of
            counts per spot will be used.
        spatial_key: Key in ``adata.obsm`` where the spatial coordinates are stored.
        control: Parameters for the Gaussian process, e.g. number of kernel components, number of inducing points.
        rng: Random number generator.

    Returns:
        A dictionary with the results. The dictionary has an additional method ``to_df``, which returns a DataFrame
        with the key results.
    """
    if not normalized and genes is None:
        warnings.warn(
            "normalized is False and no genes are given. Assuming that adata contains complete data set, will normalize and fit a GP for every gene."
        )

    if not normalized:
        adata = normalize_counts(adata, sizefactor_col, layer, copy=True)

    data = adata[:, genes] if genes is not None else adata
    X = data.obsm[spatial_key]
    counts = data.X if layer is None else data.layers[layer]

    gp = control.gp
    if gp is None:
        if data.n_obs < 1000:
            gp = GP.GPR
        else:
            gp = GP.SGPR

    results = DataSetResults()
    X = tf.convert_to_tensor(X, dtype=gpflow.config.default_float())
    t = tqdm(data.var_names)
    opt = gpflow.optimizers.Scipy()

    logging.info("Fitting gene models")
    if gp == GP.GPR:
        for g, gene in enumerate(t):
            t.set_description(gene, refresh=False)
            model = GPR(
                X,
                Y=tf.convert_to_tensor(
                    dense_slice(counts[:, g])[:, np.newaxis],
                    dtype=gpflow.config.default_float(),
                ),
                n_kernel_components=control.ncomponents,
                ard=control.ard,
            )
            results[gene] = GeneGP(model, opt.minimize, method="bfgs")
    elif gp == GP.SGPR:
        ninducers = (
            np.ceil(np.sqrt(data.n_obs)).astype(np.int32)
            if control.ninducers is None
            else control.ninducers
        )
        if control.ipm == SGPIPM.free or control.ipm == SGPIPM.random:
            inducers = X[rng.integers(0, X.shape[0], ninducers), :]
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
                    dense_slice(counts[:, g])[:, np.newaxis],
                    dtype=gpflow.config.default_float(),
                ),
                inducing_variable=inducers,
                n_kernel_components=control.ncomponents,
                ard=control.ard,
            )
            results[gene] = GeneGP(model, opt.minimize, method=method)

    logging.info("Finished fitting models to %i genes" % data.n_vars)
    return results


def fit_fast(
    adata: AnnData,
    genes: Optional[List[str]] = None,
    layer: Optional[str] = None,
    normalized: bool = False,
    sizefactor_col: Optional[str] = None,
    sparse: Optional[bool] = None,
    spatial_key: str = "spatial",
    kernel_space: Optional[Dict[str, Union[float, List[float]]]] = None,
) -> pd.DataFrame:
    """
    Fits Gaussian processes to genes.

    This uses the inflexible but fast technique of Svensson et al. (2018). In particular, the kernel lengthscale is not
    optimized, but must be given beforehand. Multiple kernel functions and lengthscales can be specified, the best-fitting
    model will be retained. To further improve speed, sparse GPs are used for large (>1000 observations) data sets with
    inducing points located on a regular grid.

    Args:
        adata: The annotated data matrix.
        genes: List of genes to base the analysis on. Defaults to all genes.
        layer: Name of the AnnData object layer to use. By default ``adata.X`` is used.
        normalized: Whether the data are already normalized to an approximately Gaussian likelihood.
            If ``False``, they will be normalized using the workflow from Svensson et al, 2018.
        sizefactor_col: Column in ``adata.obs`` to be used for normalization. If ``None``, total number of
            counts per spot will be used.
        spatial_key: Key in ``adata.obsm`` where the spatial coordinates are stored.
        sparse: Whether to use sparse GPs. Slightly faster on large datasets, but less precise.
            Defaults to sparse GPs if more than 1000 data points are given.
        kernel_space: Kernels to test against. Dictionary with the name of the kernel function as key and list of
            lengthscales (if applicable) as values. Currently, three kernel functions are known:

            * ``SE``, the squared exponential kernel :math:`k(\\boldsymbol{x}^{(1)}, \\boldsymbol{x}^{(2)}; l) = \\exp\\left(-\\frac{\\lVert \\boldsymbol{x}^{(1)} - \\boldsymbol{x}^{(2)} \\rVert}{l^2}\\right)`
            * ``PER``, the periodic kernel :math:`k(\\boldsymbol{x}^{(1)}, \\boldsymbol{x}^{(2)}; l) = \\cos\\left(2 \pi \\frac{\\sum_i (x^{(1)}_i - x^{(2)}_i)}{l}\\right)`
            * ``linear``, the linear kernel :math:`k(\\boldsymbol{x}^{(1)}, \\boldsymbol{x}^{(2)}) = (\\boldsymbol{x}^{(1)})^\\top \\boldsymbol{x}^{(2)}`

            By default, 5 squared exponential and 5 periodic kernels with lengthscales spanning the range of the
            data will be used.

    Returns:
        A Pandas DataFrame with the results.
    """
    if not normalized and genes is None:
        warnings.warn(
            "normalized is False and no genes are given. Assuming that adata contains complete data set, will normalize and fit a GP for every gene."
        )

    if not normalized:
        adata = normalize_counts(adata, sizefactor_col, layer, copy=True)

    data = adata[:, genes] if genes is not None else adata

    X = data.obsm[spatial_key]
    counts = data.X if layer is None else data.layers[layer]

    dcache = DistanceCache(X)
    if kernel_space is None:
        kernel_space = default_kernel_space(dcache)

    logging.info("Fitting gene models")
    n_models = 0
    Z = None
    if sparse is None and X.shape[0] > 1000 or sparse:
        Z = inducers_grid(X, np.maximum(100, np.sqrt(data.n_obs)))

    results = []
    for kern, kname in kspace_walk(kernel_space, dcache):
        model = model_factory(X, Z, kern)
        res = fit_model(model, data.var_names, counts)
        res["model"] = kname
        results.append(res)
        n_models += 1

    n_genes = data.n_vars
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


def fit(
    adata: AnnData,
    genes: Optional[List[str]] = None,
    layer: Optional[str] = None,
    normalized=False,
    spatial_key: str = "spatial",
    control: Optional[GPControl] = GPControl(),
    kernel_space: Optional[Dict[str, float]] = None,
    rng: np.random.Generator = np.random.default_rng(),
) -> pd.DataFrame:
    """
    Fits Gaussian processes to genes.

    This dispatches to :py:func:`fit_fast` if ``control`` is ``None``, otherwise to :py:func:`fit_detailed`.
    All arguments are forwarded.

    Returns: A Pandas DataFrame with the results.
    """
    if control is None:
        return fit_fast(adata, genes, layer, normalized, spatial_key, kernel_space)
    else:
        return (
            fit_detailed(adata, genes, layer, normalized, spatial_key, control, rng)
            .to_df(modelcol="model")
            .reset_index(drop=True)
        )
