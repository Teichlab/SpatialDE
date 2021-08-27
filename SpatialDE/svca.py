from time import time
from typing import Optional, List, Tuple, Union
import warnings
from dataclasses import dataclass, field

from tqdm.auto import tqdm, trange
from anndata import AnnData
import numpy as np
import pandas as pd
import gpflow
from gpflow.utilities import to_default_float
import tensorflow as tf
import tensorflow_probability as tfp

from ._internal.svca import SVCA, SVCAInteractionScoreTest
from ._internal.optimizer import MultiScipyOptimizer
from ._internal.util import get_l_limits, bh_adjust, calc_sizefactors, dense_slice
from ._internal.distance_cache import DistanceCache


def test_spatial_interactions(
    adata: AnnData,
    layer: Optional[str] = None,
    spatial_key: str = "spatial",
    ard: bool = False,
    sizefactors: Optional[np.ndarray] = None,
    copy: bool = False,
) -> Tuple[pd.DataFrame, Union[AnnData, None]]:
    """
    Fit an SVCA model (Arnol, 2019) and test for spatial cell-cell interactions.

    In contrast to the original publication, which used a Gaussian approximation with a
    likelihood ratio test, this fits a Poisson GLMM and uses a score test.

    Args:
        adata: The annotated data matrix.
        layer: Name of the AnnData object layer to use. By default ``adata.X`` is used.
        spatial_key: Key in ``adata.obsm`` where the spatial coordinates are stored.
        ard: Whether to use automatic relevance determination for the kernel. This amounts to
            a separate lengthscale for each spatial dimension.
        sizefactors: Scaling factors for the observations. Defaults to total read counts.
        copy: Whether to return a copy of ``adata`` with results or write the results into ``adata``
            in-place.

    Returns:
        A tuple. The first element is a Pandas DataFrame with the test results, the second is ``None``
        if ``copy == False`` or an ``AnnData`` object.
    """
    if sizefactors is None:
        sizefactors = calc_sizefactors(adata)

    X = adata.obsm[spatial_key]
    l_min, l_max = get_l_limits(DistanceCache(X))
    lscales = l_min if not ard else [l_min] * X.shape[1]
    kernel = gpflow.kernels.SquaredExponential(lengthscales=lscales)
    kernel.lengthscales.transform = tfp.bijectors.Sigmoid(
        low=to_default_float(0.5 * l_min), high=to_default_float(2 * l_max)
    )
    gpflow.set_trainable(kernel.variance, False)

    results = []
    parameters = []
    test = SVCAInteractionScoreTest(
        dense_slice(adata.X if layer is None else adata.layers[layer]), X, sizefactors, kernel
    )

    params = gpflow.utilities.parameter_dict(test.kernel[0])
    sortedkeys = sorted(params.keys())
    dtype = np.dtype([(k, params[k].dtype.as_numpy_dtype) for k in sortedkeys])

    for i, g in enumerate(tqdm(adata.var_names)):
        t0 = time()
        res, _ = test(i, None)
        t = time() - t0
        results.append({"time": t, "pval": res.pval.numpy(), "gene": g})
        params = gpflow.utilities.read_values(test.kernel[0])
        parameters.append(tuple([params[k] for k in sortedkeys]))

    results = pd.DataFrame(results)
    results.loc[
        results.pval > 1, "pval"
    ] = 1  # this seems to be a bug in tensorflow_probability, survival_function should never be >1
    results["padj"] = bh_adjust(results.pval.to_numpy())

    if copy:
        adata = adata.copy()
        toreturn = adata
    else:
        toreturn = None
    adata.varm["svca"] = np.array(parameters, dtype=dtype)
    adata.obsm["svca_sizefactors"] = sizefactors
    adata.uns["svca_ard"] = ard
    adata.uns["svca_layer"] = layer

    return results, toreturn


def fit_spatial_interactions(
    adata: AnnData,
    layer: Optional[str] = None,
    genes: Optional[List[str]] = None,
    spatial_key: str = "spatial",
    ard: bool = False,
    sizefactors: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Estimate magnitude of spatial cell-cell interactions using an SVCA model (ARnol, 2019).

    In contrast to the original publication, which used a Gaussian approximation, this fits
    a Poisson GLMM. This function is intendend to be used after :py:func:`test_spatial_interactions`
    to analyse the genes showing significant cell-cell interactions.

    Args:
        adata: The annotated data matrix.
        layer: Name of the AnnData object layer to use. By default ``adata.X`` is used.
        genes: List of genes to analyze. Defaults to all genes.
        spatial_key: Key in ``adata.obsm`` where the spatial coordinates are stored.
        ard: Whether to use automatic relevance determination for the kernel. This amounts to
            a separate lengthscale for each spatial dimension.
        sizefactors: Scaling factors for the observations. Defaults to total read counts.

    Returns: A Pandas DataFrame with the results.
    """
    if (
        "svca" not in adata.varm
        or "svca_sizefactors" not in adata.obsm
        or "svca_ard" not in adata.uns
        or "svca_layer" not in adata.uns
    ):
        warnings.warn("SVCA parameters not found in adata. Performing ab initio fitting.")
        if genes is None:
            genes = adata.var_names
        if sizefactors is None:
            sizefactors = calc_sizefactors(adata)
        trainable = True
    else:
        if genes is None:
            warnings.warn("No genes given. Fitting all genes.")
            genes = adata.var_names
        sizefactors = adata.obsm["svca_sizefactors"]
        ard = adata.uns["svca_ard"]
        trainable = False

    X = adata.obsm[spatial_key]
    l_min, l_max = get_l_limits(DistanceCache(X))
    lscales = l_min if not ard else [l_min] * X.shape[1]
    kernel = gpflow.kernels.SquaredExponential(lengthscales=lscales)
    kernel.lengthscales.transform = tfp.bijectors.Sigmoid(
        low=to_default_float(0.5 * l_min), high=to_default_float(2 * l_max)
    )
    gpflow.set_trainable(kernel.variance, False)

    model = SVCA(
        dense_slice(
            adata.X if adata.uns["svca_layer"] is None else adata.layers[adata.uns["svca_layer"]]
        ),
        X,
        sizefactors,
        kernel,
    )
    model.use_interactions(True)

    idx = np.argsort(adata.var_names)
    idx = idx[np.searchsorted(adata.var_names.to_numpy(), genes, sorter=idx)]
    if not trainable:
        param_names = adata.varm["svca"].dtype.names

    results = []
    for g, i in zip(tqdm(genes), idx):
        model.currentgene = i
        if not trainable:
            gpflow.utilities.multiple_assign(
                model.kernel, {n: v for n, v in zip(param_names, adata.varm["svca"][i])}
            )
        t0 = time()
        model.optimize()
        t = time() - t0
        fracvars = model.fraction_variance()._asdict()
        fracvars.update({"gene": g, "time": t})
        results.append(fracvars)
    return pd.DataFrame(results)
