import numpy as np
from scipy.sparse import issparse
import pandas as pd
import tensorflow as tf

from anndata import AnnData

from enum import Enum, auto
from typing import Optional, Union

import logging

from .distance_cache import DistanceCache
from ..kernels import Linear, SquaredExponential, Cosine

def get_dtype(df: pd.DataFrame, msg="Data frame"):
    dtys = df.dtypes.unique()
    if dtys.size > 1:
        logging.warning("%s has more than one dtype, selecting the first one" % msg)
    return dtys[0]

def dense_slice(slice):
    if issparse(slice):
        slice = slice.toarray()
    return np.squeeze(slice)


def bh_adjust(pvals):
    order = np.argsort(pvals)
    alpha = np.minimum(
        1, np.maximum.accumulate(len(pvals) / np.arange(1, len(pvals) + 1) * pvals[order])
    )
    return alpha[np.argsort(order)]


def calc_sizefactors(adata: AnnData):
    return adata.X.sum(axis=1).squeeze()


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


def default_kernel_space(X: np.ndarray, cache: DistanceCache):
    l_min, l_max = get_l_limits(cache)
    return {
        "SE": np.logspace(np.log10(l_min), np.log10(l_max), 5),
        "PER": np.logspace(np.log10(l_min), np.log10(l_max), 5),
    }
