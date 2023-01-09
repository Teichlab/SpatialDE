from typing import Optional, List, Union, Tuple
import warnings
from dataclasses import dataclass
from collections import namedtuple
from collections.abc import Iterable
from numbers import Real, Integral

import numpy as np
import tensorflow as tf

from gpflow import default_float, default_jitter, Parameter, set_trainable
from gpflow.utilities import to_default_float, positive
from gpflow.optimizers import Scipy

from anndata import AnnData

from ._internal.kernels import SquaredExponential
from ._internal.util import normalize_counts, get_l_limits
from ._internal.distance_cache import DistanceCache
from ._internal.util_mixture import prune_components, prune_labels


@dataclass(frozen=True)
class SpatialPatternParameters:
    """
    Parameters for automated expession histology.

    Args:
        nclasses: Maximum number of regions to consider. Defaults to the square root of the number of observations.
        lengthscales: List of kernel lenthscales. Defaults to a single lengthscale of the minimum distance between
            observations.
        pattern_prune_threshold: Probability threshold at which unused patterns are removed. Defaults to ``1e-6``.
        method: Optimization algorithm, must be known to ``scipy.optimize.minimize``. Defaults to ``l-bfgs-b``.
        tol: Convergence tolerance. Defaults to 1e-9.
        maxiter: Maximum number of iterations. Defaults to ``1000``.
        gamma_1: Parameter of the noise variance prior, defaults to ``1e-14``.
        gamma_2: Parameter of the noise variance prior, defaults to ``1e-14``.
        eta_1: Parameter of the Dirichlet process hyperprior, defaults to ``1``.
        eta_2: Parameter of the Dirichlet process hyperprior, defaults to ``1``.
    """

    nclasses: Optional[Integral] = None
    lengthscales: Optional[Union[Real, List[Real]]] = None
    pattern_prune_threshold: float = 1e-6
    method: str = "l-bfgs-b"
    tol: Optional[Real] = 1e-9
    maxiter: Integral = 1000
    gamma_1: Real = 1e-14
    gamma_2: Real = 1e-14
    eta_1: Real = 1
    eta_2: Real = 1

    def __post_init__(self):
        if self.nclasses is not None:
            assert not isinstance(
                self.lengthscales, Iterable
            ), "You must specify either nclasses or a list of lengthscales"
        if isinstance(self.lengthscales, Real):
            assert self.lengthscales > 0, "Lengthscales must be positive"
        elif self.lengthscales is not None:
            for l in self.lengthscales:
                assert l > 0, "Lengthscales must be positive"
        assert (
            self.pattern_prune_threshold >= 0 and self.pattern_prune_threshold <= 1
        ), "Class pruning threshold must be between 0 and 1"
        assert self.method in ("l-bfgs-b", "bfgs"), "Method must be either bfgs or l-bfgs-b"
        if self.tol is not None:
            assert self.tol > 0, "Tolerance must be greater than 0"
        assert self.maxiter >= 1, "Maximum number of iterations must greater than or equal to 1"
        assert self.gamma_1 > 0, "Gamma1 hyperparameter must be positive"
        assert self.gamma_2 > 0, "Gamma2 hyperparameter must be positive"
        assert self.eta_1 > 0, "Eta1 hyperparameter must be positive"
        assert self.eta_2 > 0, "Eta2 hyperparameter must be positive"


@dataclass(frozen=True)
class SpatialPatterns:
    """
    Results of automated expression histology.

    Args:
        converged: Whether the optimization converged.
        status: Status of the optimization.
        labels: The estimated region labels.
        pattern_probabilities: N_obs x N_patterns array with the estimated region probabilities for each observation.
        niter: Number of iterations for the optimization.
        elbo_trace: ELBO values at each iteration.
    """

    converged: bool
    status: str
    labels: np.ndarray
    pattern_probabilities: np.ndarray
    patterns: np.ndarray
    niter: int
    elbo_trace: np.ndarray


class _SpatialPatterns(tf.Module):
    def __init__(
        self,
        X: np.ndarray,
        counts: np.ndarray,
        nclasses: Integral,
        lengthscales: List[Real],
        gamma_1: Real,
        gamma_2: Real,
        eta_1: Real,
        eta_2: Real,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        self.X = to_default_float(X)
        self.counts = to_default_float(counts)
        self.nsamples, self.ngenes = tf.shape(counts)
        self._fnsamples, self._fngenes = to_default_float(tf.shape(counts))
        self.nclasses = nclasses
        self._fnclasses = to_default_float(self.nclasses)

        self.gamma_1 = to_default_float(gamma_1)
        self.gamma_2 = to_default_float(gamma_2)
        self.eta_1 = to_default_float(eta_1)
        self.eta_2 = to_default_float(eta_2)

        dcache = DistanceCache(X)
        if lengthscales is None:
            l_min, l_max = get_l_limits(dcache)
            lengthscales = [0.5 * l_min]
        elif not isinstance(lengthscales, Iterable):
            lengthscales = [lengthscales]
        self.kernels = []
        Kernel = namedtuple("DecomposedKernel", "Lambda U")

        lcounts = np.unique(lengthscales, return_counts=True)
        for l, c in zip(*lcounts):
            k = SquaredExponential(dcache, lengthscale=l).K()
            S, U = tf.linalg.eigh(k)
            self.kernels.extend([Kernel(S, U)] * c)
        if len(self.kernels) == 1:
            self.kernels = self.kernels * nclasses

        self.phi = Parameter(
            rng.uniform(-0.01, 0.01, (self.ngenes, self.nclasses)), dtype=default_float()
        )  # to break ties, otherwise all gradients are the same
        self.etahat_2 = Parameter(
            eta_2 + default_jitter(), dtype=default_float(), transform=positive(lower=eta_2)
        )
        self.gammahat_2 = Parameter(
            self.gammahat_1,
            dtype=default_float(),
            transform=positive(lower=gamma_2),
        )

    @property
    def etahat_1(self):
        return self.eta_1 + self._fnclasses - 1

    @property
    def pihat(self):
        return tf.nn.softmax(self.phi, axis=1)

    @property
    def gammahat_1(self):
        return self.gamma_1 + 0.5 * self._fnsamples * self._fngenes

    @property
    def _sigmahat(self):
        return self.gammahat_1 / self.gammahat_2

    def Sigma_hat_inv(self, c=0):
        k = self.kernels[c]
        EigUt = k.U * ((self._sigmahat * self._N(c) * k.Lambda + 1) / k.Lambda)[tf.newaxis, :]
        return tf.matmul(k.U, EigUt, transpose_b=True)

    def Sigma_hat(self, c=0):
        k = self.kernels[c]
        EigUt = k.U * (k.Lambda / (self._sigmahat * self._N(c) * k.Lambda + 1))[tf.newaxis, :]
        return tf.matmul(k.U, EigUt, transpose_b=True)

    @property
    def mu_hat(self):
        ybar = self._ybar()
        return tf.stack(
            [self._mu_hat(c, ybar=ybar[:, c]) for c in tf.range(self.nclasses)],
            axis=1,
        )

    def _mu_hat(self, c=None, Sigma_hat=None, ybar=None):
        assert c is not None or Sigma_hat is not None and ybar is not None
        if Sigma_hat is None:
            Sigma_hat = self.Sigma_hat(c)
        if ybar is None:
            ybar = self._ybar(c)
        return self._sigmahat * tf.tensordot(Sigma_hat, ybar, axes=(-1, -1))

    def _ybar(self, c=None):
        if c is None:
            return self.counts @ self.pihat
        else:
            return tf.tensordot(self.counts, self.pihat[:, c], axes=(-1, -1))

    def _N(self, c=None):
        if c is None:
            return tf.reduce_sum(self.pihat, axis=0)
        else:
            return tf.reduce_sum(self.pihat[:, c])

    @property
    def _lhat(self):
        return tf.math.log(self.gammahat_2) - tf.math.digamma(self.gammahat_1)

    @property
    def _alphahat(self):
        return self.etahat_1 / self.etahat_2

    @property
    def _alphahat1(self):
        return 1 + self._N()[:-1]

    @property
    def _alphahat2(self):
        pihat_cumsum = tf.cumsum(tf.reduce_sum(self.pihat, axis=0), reverse=True)
        return pihat_cumsum[1:] + self._alphahat

    @property
    def _vhat2(self):
        return tf.math.digamma(self._alphahat1) - tf.math.digamma(self._alphahat1 + self._alphahat2)

    @property
    def _vhat3(self):
        return tf.math.digamma(self._alphahat2) - tf.math.digamma(self._alphahat1 + self._alphahat2)

    def elbo(self):
        pihat = self.pihat
        dotcounts = tf.reduce_sum(tf.square(self.counts), axis=0)
        sigmahat = self._sigmahat
        ybar = self._ybar()
        lhat = self._lhat
        N = self._N()

        term1 = 0.5 * (self._fnsamples * self._fngenes * lhat + sigmahat * tf.reduce_sum(dotcounts))

        term2 = 0
        for c in range(self.nclasses):
            k = self.kernels[c]
            UTybar = tf.tensordot(k.U, ybar[:, c], axes=(0, 0))
            Lambdahat = sigmahat * N[c] * k.Lambda + 1

            ybar_mu = tf.square(sigmahat) * tf.tensordot(
                UTybar, k.Lambda / Lambdahat * UTybar, axes=(-1, -1)
            )
            pimu = N[c] * (
                sigmahat**3
                * tf.tensordot(UTybar * tf.square(k.Lambda / Lambdahat), UTybar, axes=(-1, -1))
                + tf.reduce_sum(k.Lambda / Lambdahat)
            )
            muinv_ybar = tf.square(sigmahat) * tf.tensordot(
                UTybar * k.Lambda / tf.square(Lambdahat), UTybar, axes=(-1, -1)
            )
            trace_inv = tf.reduce_sum(1 / Lambdahat)
            logdet = tf.reduce_sum(tf.math.log(Lambdahat))

            term2 += ybar_mu - 0.5 * (pimu + muinv_ybar + trace_inv + logdet)

        term3 = tf.reduce_sum(N[:-1] * self._vhat2)

        vhat3_cumsum = tf.concat(((0,), tf.cumsum(self._vhat3)), axis=0)
        term4 = tf.reduce_sum(N * vhat3_cumsum)

        term5 = tf.reduce_sum((self._alphahat - self._alphahat2) * self._vhat3)
        term6 = tf.reduce_sum(pihat * self.phi)

        term7 = tf.reduce_sum(tf.reduce_logsumexp(self.phi, axis=1))
        term8 = (
            -self.gammahat_1 * (1 - tf.math.log(self.gammahat_2))
            + tf.math.lgamma(self.gammahat_1)
            + (self.gammahat_1 + 1) * lhat
        )
        term9 = tf.reduce_sum((self._alphahat1 - 1) * self._vhat2)
        term10 = tf.reduce_sum(tf.math.lbeta(tf.stack((self._alphahat1, self._alphahat2), axis=1)))
        term11 = (
            -(self.eta_1 + self._fnclasses) * tf.math.log(self.etahat_2)
            + (self.etahat_2 - self.eta_2) * self._alphahat
        )

        elbo = (
            -term1 + term2 + term3 + term4 + term5 - term6 + term7 + term8 - term9 + term10 + term11
        ) / (self._fnsamples * self._fnclasses * self._fngenes)
        return elbo


def spatial_patterns(
    adata: AnnData,
    genes: Optional[List[str]] = None,
    normalized=False,
    spatial_key="spatial",
    layer: Optional[str] = None,
    params: SpatialPatternParameters = SpatialPatternParameters(),
    rng: np.random.Generator = np.random.default_rng(),
    copy: bool = False,
) -> Tuple[SpatialPatterns, Union[AnnData, None]]:
    """
    Detect spatial patterns of gene expression and assign genes to patterns.

    Uses a Gaussian process mixture. A Dirichlet process prior allows
    to automatically determine the number of distinct regions in the dataset.

    Args:
        adata: The annotated data matrix.
        genes: List of genes to base the analysis on. Defaults to all genes.
        normalized: Whether the data are already normalized to an approximately Gaussian likelihood.
            If ``False``, they will be normalized using the workflow from Svensson et al, 2018.
        spatial_key: Key in ``adata.obsm`` where the spatial coordinates are stored.
        layer: Name of the AnnData object layer to use. By default ``adata.X`` is used.
        params: Parameters for the algorithm, e.g. prior distributions, spatial smoothness, etc.
        rng: Random number generator.
        copy: Whether to return a copy of ``adata`` with results or write the results into ``adata``
            in-place.

    Returns:
        A tuple. The first element is a :py:class:`SpatialPatterns`, the second is ``None`` if ``copy == False``
        or an ``AnnData`` object. Patterns will be in ``adata.obs["spatial_pattern_0"]``, ...,
        ``adata.obs["spatial_pattern_n"]``.
    """
    if not normalized and genes is None:
        warnings.warn(
            "normalized is False and no genes are given. Assuming that adata contains complete data set, will normalize and fit a GP for every gene."
        )
    data = normalize_counts(adata, copy=True) if not normalized else adata
    if genes is not None:
        data = data[:, genes]

    X = data.obsm[spatial_key]
    counts = data.X if layer is None else adata.layers[layer]

    # This is important, we only care about co-expression, not absolute levels.
    counts = counts - tf.reduce_mean(counts, axis=0)
    counts = counts / tf.math.reduce_std(counts, axis=0)

    nclasses = params.nclasses
    if nclasses is None:
        if isinstance(params.lengthscales, Iterable):
            nclasses = len(params.lengthscales)
        else:
            nclasses = int(np.ceil(np.sqrt(data.n_vars)))

    patterns = _SpatialPatterns(
        X,
        counts,
        nclasses,
        params.lengthscales,
        params.gamma_1,
        params.gamma_2,
        params.eta_1,
        params.eta_2,
        rng,
    )
    opt = Scipy()
    elbo_trace = [patterns.elbo()]
    res = opt.minimize(
        lambda: -patterns.elbo(),
        patterns.trainable_variables,
        method=params.method,
        step_callback=lambda step, vars, vals: elbo_trace.append(patterns.elbo()),
        tol=params.tol,
        options={"maxiter": params.maxiter},
    )

    prune_threshold = tf.convert_to_tensor(params.pattern_prune_threshold, dtype=default_float())
    idx, labels = prune_components(
        tf.argmax(patterns.pihat, axis=1),
        tf.transpose(patterns.pihat),
        prune_threshold,
        everything=True,
    )
    pihat = tf.linalg.normalize(tf.gather(patterns.pihat, idx, axis=1), ord=1, axis=1)[0]
    patterns = tf.gather(patterns.mu_hat, idx, axis=1).numpy()

    if copy:
        adata = adata.copy()
        toreturn = adata
    else:
        toreturn = None
    for i in range(patterns.shape[1]):
        adata.obs[f"spatial_pattern_{i}"] = patterns[:, i]

    return (
        SpatialPatterns(
            converged=res.success,
            status=res.message,
            labels=labels.numpy(),
            pattern_probabilities=pihat.numpy(),
            patterns=patterns,
            niter=res.nit,
            elbo_trace=np.asarray(elbo_trace),
        ),
        toreturn,
    )
