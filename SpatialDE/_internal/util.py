import numpy as np
from scipy.sparse import issparse
import scipy.stats
import pandas as pd
import tensorflow as tf

import NaiveDE
from anndata import AnnData

from enum import Enum, auto
from typing import Optional, Union

import logging

from .distance_cache import DistanceCache
from .kernels import Linear, SquaredExponential, Cosine


def get_dtype(df: pd.DataFrame, msg="Data frame"):
    dtys = df.dtypes.unique()
    if dtys.size > 1:
        logging.warning("%s has more than one dtype, selecting the first one" % msg)
    return dtys[0]


def normalize_counts(
    adata: AnnData,
    sizefactorcol: Optional[str] = None,
    layer: Optional[str] = None,
    copy: bool = False,
):
    if copy:
        adata = adata.copy()

    if sizefactorcol is None:
        sizefactors = pd.DataFrame({"sizefactors": calc_sizefactors(adata, layer=layer)})
        sizefactorcol = "np.log(sizefactors)"
    else:
        sizefactors = adata.obs
    X = adata.X if layer is None else adata.layers[layer]
    stabilized = NaiveDE.stabilize(dense_slice(X.T))
    regressed = np.asarray(NaiveDE.regress_out(sizefactors, stabilized, sizefactorcol).T)
    if layer is None:
        adata.X = regressed
    else:
        adata.layers[layer] = regressed
    return adata


def dense_slice(slice):
    if issparse(slice):
        slice = slice.toarray()
    else:
        slice = np.asarray(slice)  # work around anndata.ArrayView
    return np.squeeze(slice)


def bh_adjust(pvals):
    order = np.argsort(pvals)
    alpha = np.minimum(
        1, np.maximum.accumulate(len(pvals) / np.arange(1, len(pvals) + 1) * pvals[order])
    )
    return alpha[np.argsort(order)]


def calc_sizefactors(adata: AnnData, layer=None):
    X = adata.X if layer is None else adata.layers[layer]
    return np.asarray(X.sum(axis=1)).squeeze()


def get_l_limits(cache: DistanceCache):
    R2 = cache.squaredEuclideanDistance
    R2 = R2[R2 > 1e-8]

    l_min = tf.sqrt(tf.reduce_min(R2)) * 2.0
    l_max = tf.sqrt(tf.reduce_max(R2))

    return l_min, l_max


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


def default_kernel_space(cache: DistanceCache):
    l_min, l_max = get_l_limits(cache)
    return {
        "SE": np.logspace(np.log10(l_min), np.log10(l_max), 5),
        "PER": np.logspace(np.log10(l_min), np.log10(l_max), 5),
    }


def concat_tensors(tens):
    return tf.concat([tf.reshape(t, (-1,)) for t in tens], axis=0)


def assign_concat(x, vars):
    offset = 0
    for v in vars:
        newval = tf.reshape(x[offset : (offset + tf.size(v))], v.shape)
        v.assign(newval)
        offset += tf.size(v)


def gower_factor(mat, varcomp=1):
    """Gower normalization factor for covariance matric K

    Based on https://github.com/PMBio/limix/blob/master/limix/utils/preprocess.py
    """
    return (
        varcomp
        * (tf.linalg.trace(mat) - tf.reduce_sum(tf.reduce_mean(mat, axis=0)))
        / tf.cast(tf.shape(mat)[0] - 1, mat.dtype)
    )


def quantile_normalize(mat):
    idx = np.argsort(mat, axis=0) + 0.5
    return scipy.stats.norm(loc=0, scale=1).ppf(idx / mat.shape[0])
