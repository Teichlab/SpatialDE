from typing import Optional, List, Union, Tuple
from dataclasses import dataclass
from enum import Enum, auto
import logging
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from gpflow.utilities.ops import square_distance

from anndata import AnnData

from ._internal.util import calc_sizefactors, dense_slice
from ._internal.util_mixture import prune_components, prune_labels


@dataclass(frozen=True)
class TissueSegmentationParameters:
    """
    Parameters for tissue segmentation.

    Args:
        nclasses: Maximum number of regions to consider. Defaults to the square root of the number of observations.
        neighbors: Number of neighbors for the nearest-neighbor graph. Defaults to a fully connected graph (there is
            no speed difference). A value of 0 makes the model ignore spatial information and reduces it to a Poisson
            mixture model with a Dirichlet process prior.
        smoothness_factor: Spatial smoothness. Larger values induce more fine-grained segmentations. This value will
            be multiplied with the minimum squared distance within the data set, so it is dimensionless. Defaults to ``2``.
        class_prune_threshold: Probability threshold at which unused regions are removed. Defaults to ``1e-6``.
        abstol: Absolute convergence tolerance. Defaults to ``1e-12``.
        reltol: Relative convergence tolerance. Defaults to ``1e-12``.
        maxiter: Maximum number of iterations. Defaults to ``1000``.
        gamma_1: Parameter of the Poisson mean prior, defaults to ``1e-14``.
        gamma_2: Parameter of the Poisson mean prior, defaults to ``1e-14``.
        eta_1: Parameter of the Dirichlet process hyperprior, defaults to ``1``.
        eta_2: Parameter of the Dirichlet process hyperprior, defaults to ``1``.
    """

    nclasses: Optional[int] = None
    neighbors: Optional[int] = None
    smoothness_factor: float = 2
    class_prune_threshold: float = 1e-6
    abstol: float = 1e-12
    reltol: float = 1e-12
    maxiter: int = 1000
    gamma_1: float = 1e-14
    gamma_2: float = 1e-14
    eta_1: float = 1
    eta_2: float = 1

    def __post_init__(self):
        assert (
            self.nclasses is None or self.nclasses >= 1
        ), "Number of classes must be None or at least 1"
        assert (
            self.neighbors is None or self.neighbors >= 0
        ), "Number of neighbors must be None or at least 0"
        assert self.smoothness_factor > 0, "Smoothness factor must be greater than 0"
        assert (
            self.class_prune_threshold >= 0 and self.class_prune_threshold <= 1
        ), "Class pruning threshold must be between 0 and 1"
        assert self.abstol > 0, "Absolute tolerance must be greater than 0"
        assert self.reltol > 0, "Relative tolerance must be greater than 0"
        assert self.maxiter >= 1, "Maximum number of iterations must be greater than or equal to 1"
        assert self.gamma_1 > 0, "Gamma1 hyperparameter must be greater than 0"
        assert self.gamma_2 > 0, "Gamma2 hyperparameter must be greater than 0"
        assert self.eta_1 > 0, "Eta1 hyperparameter must be greater than 0"
        assert self.eta_2 > 0, "Eta2 hyperparameter must be greater than 0"


class TissueSegmentationStatus(Enum):
    AbsoluteToleranceReached = auto()
    RelativeToleranceReached = auto()
    MaximumIterationsReached = auto()


@dataclass(frozen=True)
class TissueSegmentation:
    """
    Results of tissue segmentation.

    Args:
        converged: Whether the optimization converged.
        status: Status of the optimization.
        labels: The estimated region labels.
        class_probabilities: N_obs x N_regions array with the estimated region probabilities for each observation.
        gammahat_1: N_classes x N_genes array with the estimated parameter of the gene expression posterior.
        gammahat_2: N_classes x 1 array with the estimated parameter of the gene expression posterior.
        niter: Number of iterations for the optimization.
        prune_iterations: Iterations at which unneeded regions were removed.
        elbo_trace: ELBO values at each iteration.
        nclasses_trace: Number of regions at each iteration.
    """

    converged: bool
    status: TissueSegmentationStatus
    labels: np.ndarray
    class_probabilities: np.ndarray
    gammahat_1: np.ndarray
    gammahat_2: np.ndarray
    niter: int
    prune_iterations: np.ndarray
    elbo_trace: np.ndarray
    nclasses_trace: np.ndarray


@tf.function(experimental_relax_shapes=True)
def _segment(
    counts: tf.Tensor,
    sizefactors: tf.Tensor,
    distances: tf.Tensor,
    nclasses: tf.Tensor,
    fnclasses: tf.Tensor,
    ngenes: tf.Tensor,
    labels: tf.Tensor,
    gamma_1: tf.Tensor,
    gamma_2: tf.Tensor,
    eta_1: tf.Tensor,
    eta_2: tf.Tensor,
    alphahat_1: tf.Tensor,
    alphahat_2: tf.Tensor,
    etahat_2: tf.Tensor,
    gammahat_1: tf.Tensor,
    gammahat_2: tf.Tensor,
):
    eta_1_nclasses = eta_1 + fnclasses
    if labels is not None and distances is not None:
        p_x_neighborhood = tf.TensorArray(counts.dtype, size=tf.shape(gammahat_1)[0])
        for c in tf.range(tf.shape(gammahat_1)[0]):
            p_x_neighborhood = p_x_neighborhood.write(
                c, -tf.reduce_sum(tf.where(labels != c, distances, 0), axis=1)
            )
        p_x_neighborhood = p_x_neighborhood.stack()
        p_x_neighborhood = p_x_neighborhood - tf.reduce_logsumexp(
            p_x_neighborhood, axis=0, keepdims=True
        )
    else:
        p_x_neighborhood = tf.convert_to_tensor(0, counts.dtype)

    lambdahat_1 = gammahat_1 / gammahat_2
    lambdahat_2 = tf.math.digamma(gammahat_1) - tf.math.log(gammahat_2)
    alpha12 = alphahat_1 + alphahat_2
    dgalpha = tf.math.digamma(alpha12)
    vhat2 = tf.math.digamma(alphahat_1) - dgalpha
    vhat3 = tf.math.digamma(alphahat_2) - dgalpha
    alphahat = (eta_1_nclasses - 1) / etahat_2
    vhat3_cumsum = tf.cumsum(vhat3) - vhat3

    vhat_sum = (tf.concat(((0,), vhat3_cumsum), axis=0) + tf.concat((vhat2, (0,)), axis=0))[
        :, tf.newaxis
    ]

    phi = (
        p_x_neighborhood
        + vhat_sum
        + tf.matmul(lambdahat_2, counts, transpose_b=True)
        - sizefactors * tf.reduce_sum(lambdahat_1, axis=1, keepdims=True)
    )
    pihat = tf.nn.softmax(phi, axis=0)
    pihat_cumsum = tf.cumsum(pihat, axis=0, reverse=True) - pihat

    vhat3_sum = tf.reduce_sum(vhat3)
    gammahat_1 = gamma_1 + pihat @ counts
    gammahat_2 = gamma_2 + tf.matmul(pihat, sizefactors, transpose_b=True)
    etahat_2 = eta_2 - vhat3_sum
    alphahat_1 = 1 + ngenes * tf.reduce_sum(pihat, axis=1)[:-1]
    alphahat_2 = ngenes * tf.reduce_sum(pihat_cumsum, axis=1)[:-1] + alphahat

    elbo = (
        tf.reduce_sum(pihat * p_x_neighborhood)
        + tf.reduce_sum(
            pihat
            * (
                vhat_sum
                + tf.matmul(lambdahat_2, counts, transpose_b=True)
                - tf.reduce_sum(lambdahat_1, axis=1, keepdims=True) * sizefactors
            )
        )
        + tf.reduce_sum((alphahat - alphahat_2) * vhat3)
        + tf.reduce_sum((gamma_1 - gammahat_1) * lambdahat_2)
        + tf.reduce_sum((gammahat_2 - gamma_2) * lambdahat_1)
        - tf.reduce_sum(pihat * phi)
        + tf.reduce_sum(tf.reduce_logsumexp(phi, axis=0))
        - tf.reduce_sum(gammahat_1 * tf.math.log(gammahat_2) - tf.math.lgamma(gammahat_1))
        - tf.reduce_sum((alphahat_1 - 1) * vhat2)
        + tf.reduce_sum(tf.math.lbeta(tf.stack((alphahat_1, alphahat_2), axis=1)))
        - eta_1_nclasses * tf.math.log(etahat_2)
        + (etahat_2 - eta_2) * alphahat
    ) / tf.cast(nclasses * ngenes * tf.shape(counts)[0], counts.dtype)

    return pihat, alphahat_1, alphahat_2, etahat_2, gammahat_1, gammahat_2, elbo


def tissue_segmentation(
    adata: AnnData,
    layer: Optional[str] = None,
    genes: Optional[List[str]] = None,
    sizefactors: Optional[np.ndarray] = None,
    spatial_key: str = "spatial",
    params: TissueSegmentationParameters = TissueSegmentationParameters(),
    labels: Optional[Union[np.ndarray, tf.Tensor]] = None,
    rng: np.random.Generator = np.random.default_rng(),
    copy=False,
) -> Tuple[TissueSegmentation, Union[AnnData, None]]:
    """
    Segment a spatial transcriptomics dataset into distinct spatial regions.

    Uses a hidden Markov random field (HMRF) model with a Poisson likelihood. A Dirichlet process prior allows
    to automatically determine the number of distinct regions in the dataset.

    Args:
        adata: The annotated data matrix.
        layer: Name of the AnnData object layer to use. By default ``adata.X`` is used.
        genes: List of genes to base the segmentation on. Defaults to all genes.
        sizefactors: Scaling factors for the observations. Defaults to total read counts.
        spatial_key: Key in ``adata.obsm`` where the spatial coordinates are stored.
        params: Parameters for the algorithm, e.g. prior distributions, spatial smoothness, etc.
        labels: Initial label for each observation. Defaults to a random initialization.
        rng: Random number generator.
        copy: Whether to return a copy of ``adata`` with results or write the results into ``adata``
            in-place.

    Returns:
        A tuple. The first element is a :py:class:`TissueSegmentation`, the second is ``None`` if ``copy == False``
        or an ``AnnData`` object. Region labels will be in ``adata.obs["segmentation_labels"]`` and region
        probabilities in ``adata.obsm["segmentation_class_probabilities"]``.
    """
    if genes is None and sizefactors is None:
        warnings.warn(
            "Neither genes nor sizefactors are given. Assuming that adata contains complete data set, will calculate size factors and perform segmentation using the complete data set.",
            RuntimeWarning,
        )

    if sizefactors is None:
        sizefactors = calc_sizefactors(adata, layer=layer)
    if genes is not None:
        ngenes = len(genes)
        data = adata[:, genes]
    else:
        ngenes = adata.n_vars
        data = adata
    try:
        X = data.obsm[spatial_key]
    except KeyError:
        X = None

    dtype = tf.float64
    labels_dtype = tf.int32
    nclasses = params.nclasses
    nsamples = data.n_obs
    if nclasses is None:
        nclasses = tf.cast(
            tf.math.ceil(tf.sqrt(tf.convert_to_tensor(nsamples, dtype=tf.float32))), tf.int32
        )
    fngenes = tf.cast(ngenes, dtype=dtype)
    fnclasses = tf.cast(nclasses, dtype=dtype)

    sizefactors = tf.convert_to_tensor(sizefactors[np.newaxis, :], dtype=dtype)

    gamma_1 = tf.convert_to_tensor(params.gamma_1, dtype=dtype)
    gamma_2 = tf.convert_to_tensor(params.gamma_2, dtype=dtype)
    eta_1 = tf.convert_to_tensor(params.eta_1, dtype=dtype)
    eta_2 = tf.convert_to_tensor(params.eta_2, dtype=dtype)

    counts = tf.convert_to_tensor(
        dense_slice(data.X if layer is None else data.layers[layer]), dtype=dtype
    )

    distances = None
    if X is not None and (params.neighbors is None or params.neighbors > 0):
        X = tf.convert_to_tensor(X, dtype=dtype)
        distances = square_distance(X, None)
        if params.neighbors is not None and params.neighbors < nsamples:
            distances, indices = tf.math.top_k(-distances, k=params.neighbors + 1, sorted=True)
            distances = -distances[:, 1:]
            distances = 2 * params.smoothness_factor * tf.reduce_min(distances) / distances
            indices = indices[:, 1:]
            indices = tf.stack(
                (
                    tf.repeat(tf.range(distances.shape[0]), indices.shape[1]),
                    tf.reshape(indices, -1),
                ),
                axis=1,
            )
            dists = tf.reshape(distances, -1)
            distances = tf.scatter_nd(indices, dists, (distances.shape[0], distances.shape[0]))
            distances = tf.tensor_scatter_nd_update(
                distances, indices[:, ::-1], dists
            )  # symmetrize
        else:
            distances = tf.linalg.set_diag(
                distances, tf.repeat(tf.convert_to_tensor(np.inf, dtype), tf.shape(distances)[0])
            )
            distances = 2 * params.smoothness_factor * tf.reduce_min(distances) / distances
    else:
        logging.info("Not using spatial information, fitting Poisson mixture model instead.")

    if labels is not None:
        labels = tf.squeeze(tf.convert_to_tensor(labels, dtype=labels_dtype))
        if tf.rank(labels) > 1 or tf.shape(labels)[0] != nsamples:
            labels = None
            warnings.warn(
                "Shape of given labels does not conform to data. Initializing with random labels.",
                RuntimeWarning,
            )
    if labels is None and distances is not None:
        labels = tf.convert_to_tensor(rng.choice(nclasses, nsamples), dtype=labels_dtype)

    alphahat_1 = tf.ones(shape=(nclasses - 1,), dtype=dtype)
    alphahat_2 = tf.ones(shape=(nclasses - 1,), dtype=dtype)
    etahat_2 = eta_1 + fnclasses - 1
    gammahat_1 = tf.fill((nclasses, ngenes), tf.convert_to_tensor(1e-6, dtype=dtype))
    gammahat_2 = tf.fill((nclasses, 1), tf.convert_to_tensor(1e-6, dtype=dtype))

    prune_threshold = tf.convert_to_tensor(params.class_prune_threshold, dtype=dtype)
    lastelbo = -tf.convert_to_tensor(np.inf, dtype=dtype)
    elbos = []
    nclassestrace = []
    pruneidx = []
    converged = False
    status = TissueSegmentationStatus.MaximumIterationsReached
    for i in range(params.maxiter):
        (pihat, alphahat_1, alphahat_2, etahat_2, gammahat_1, gammahat_2, elbo,) = _segment(
            counts,
            sizefactors,
            distances,
            nclasses,
            fnclasses,
            ngenes,
            labels,
            gamma_1,
            gamma_2,
            eta_1,
            eta_2,
            alphahat_1,
            alphahat_2,
            etahat_2,
            gammahat_1,
            gammahat_2,
        )
        labels = tf.math.argmax(pihat, axis=0, output_type=labels_dtype)
        elbos.append(elbo.numpy())
        nclassestrace.append(nclasses)
        elbodiff = tf.abs(elbo - lastelbo)
        if elbodiff < params.abstol:
            converged = True
            status = TissueSegmentationStatus.AbsoluteToleranceReached
            break
        elif elbodiff / tf.minimum(tf.abs(elbo), tf.abs(lastelbo)) < params.reltol:
            converged = True
            status = TissueSegmentationStatus.AbsoluteToleranceReached
            break
        lastelbo = elbo

        if i == 1 or not i % 10:
            idx, labels = prune_components(labels, pihat, prune_threshold, everything=True)
            if tf.size(idx) < tf.shape(gammahat_1)[0]:
                pruneidx.append(i)
            alphahat_1 = tf.gather(alphahat_1, idx[:-1], axis=0)
            alphahat_2 = tf.gather(alphahat_2, idx[:-1], axis=0)
            gammahat_1 = tf.gather(gammahat_1, idx, axis=0)
            gammahat_2 = tf.gather(gammahat_2, idx, axis=0)
            nclasses = tf.size(idx)

    idx, labels = prune_components(labels, pihat, prune_threshold, everything=True)
    pihat = tf.linalg.normalize(tf.gather(pihat, idx, axis=0), ord=1, axis=0)[0]
    gammahat_1 = tf.gather(gammahat_1, idx, axis=0)
    gammahat_2 = tf.gather(gammahat_2, idx, axis=0)

    if copy:
        adata = adata.copy()
        toreturn = adata
    else:
        toreturn = None
    labels = labels.numpy()
    pihat = pihat.numpy().T
    adata.obs["segmentation_labels"] = pd.Categorical(labels)
    adata.obsm["segmentation_class_probabilities"] = pihat
    return (
        TissueSegmentation(
            converged,
            status,
            labels,
            pihat,
            gammahat_1.numpy(),
            gammahat_2.numpy(),
            i,
            np.asarray(pruneidx),
            np.asarray(elbos),
            np.asarray(nclassestrace),
        ),
        toreturn,
    )
