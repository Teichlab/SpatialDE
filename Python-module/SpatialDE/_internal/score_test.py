from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, fields
from typing import Optional, Union, List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
from scipy.optimize import minimize

from ..kernels import Kernel

from enum import Enum, auto
import math


@dataclass(frozen=True)
class ScoreTestResults:
    kappa: Union[float, tf.Tensor]
    nu: Union[float, tf.Tensor]
    U_tilde: Union[float, tf.Tensor]
    e_tilde: Union[float, tf.Tensor]
    I_tilde: Union[float, tf.Tensor]
    pval: Union[float, tf.Tensor]

    def to_dict(self):
        ret = {}
        for f in fields(self):
            obj = getattr(self, f.name)
            if tf.is_tensor(obj):
                obj = obj.numpy()
            ret[f.name] = obj
        return ret


def combine_pvalues(
    results: Union[ScoreTestResults, List[ScoreTestResults], tf.Tensor, np.ndarray]
) -> float:
    if isinstance(results, ScoreTestResults):
        pvals = results.pval
    elif isinstance(results, list):
        pvals = tf.stack([r.pval for r in results], axis=0)
    elif tf.is_tensor(results):
        pvals = results
    elif isinstance(results, np.ndarray):
        pvals = tf.convert_to_tensor(pvals)
    else:
        raise TypeError("Unknown type for results.")

    comb = tf.reduce_mean(tf.tan((0.5 - pvals) * math.pi))
    return tfd.Cauchy(
        tf.convert_to_tensor(0, comb.dtype), tf.convert_to_tensor(1, comb.dtype)
    ).survival_function(comb)


class ScoreTest(metaclass=ABCMeta):
    dtype = tf.float64  # use single precision for better performance

    @dataclass
    class NullModel(metaclass=ABCMeta):
        pass

    def __init__(
        self, omnibus: bool = False, kernel: Optional[Union[Kernel, List[Kernel]]] = None,
    ):
        self.omnibus = omnibus

        self.n = None

        if kernel is not None:
            self.kernel = kernel

    def __call__(
        self, y: tf.Tensor, nullmodel: Optional[NullModel] = None
    ) -> Tuple[ScoreTestResults, NullModel]:
        y = tf.squeeze(y)
        if self.yidx is not None:
            y = tf.gather(y, self.yidx)
        if y.dtype is not self.dtype:
            raise TypeError(
                f"Value vector has wrong dtype. Expected: {repr(self.dtype)}, given: {repr(rawy.dtype)}"
            )
        if nullmodel is None:
            nullmodel = self._fit_null(y)
        stat, e_tilde, I_tau_tau = self._test(y, nullmodel)
        return self._calc_test(stat, e_tilde, I_tau_tau), nullmodel

    @property
    def kernel(self) -> List[Kernel]:
        return self.kernel

    @kernel.setter
    def kernel(self, kernel: Union[Kernel, List[Kernel]]):
        self._kernel = [kernel] if isinstance(kernel, Kernel) else kernel
        if len(self._kernel) > 1:
            if self.omnibus:
                self._K = self._kernel[0].K()
                for k in self._kernel[1:]:
                    self._K += k.K()
            else:
                self._K = tf.stack([k.K() for k in kernel], axis=0)
        else:
            self._K = self._kernel[0].K()
        self._K = tf.cast(self._K, self.dtype)
        self.n = tf.shape(self._K)[0]

    @abstractmethod
    def _test(
        self, y: tf.Tensor, nullmodel: NullModel
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        pass

    @abstractmethod
    def _fit_null(self, y: tf.Tensor) -> NullModel:
        pass

    @staticmethod
    def _calc_test(stat, e_tilde, I_tau_tau) -> ScoreTestResults:
        kappa = I_tau_tau / (2 * e_tilde)
        nu = 2 * e_tilde ** 2 / I_tau_tau
        pval = tfd.Chi2(nu).survival_function(stat / kappa)
        return ScoreTestResults(kappa, nu, stat, e_tilde, I_tau_tau, pval)


class NegativeBinomialScoreTest(ScoreTest):
    @dataclass
    class NullModel(ScoreTest.NullModel):
        mu: tf.Tensor
        alpha: tf.Tensor

    def __init__(
        self,
        sizefactors: tf.Tensor,
        omnibus: bool = False,
        kernel: Optional[Union[Kernel, List[Kernel]]] = None,
    ):
        self.sizefactors = tf.squeeze(
            tf.convert_to_tensor(sizefactors, dtype=tf.float64)
        )  # we want float64 here, this greatly reduces the number of iterations for the MLE
        if tf.rank(self.sizefactors) > 1:
            raise ValueError("Size factors vector must have rank 1")

        self._parameters_cache = {}
        yidx = tf.cast(tf.squeeze(tf.where(self.sizefactors > 0)), tf.int32)
        self.yidx = None

        if tf.shape(yidx)[0] != tf.shape(self.sizefactors)[0]:
            self.yidx = yidx
            self.sizefactors = tf.gather(self.sizefactors, self.yidx)
        super().__init__(omnibus, kernel)

    @ScoreTest.kernel.setter
    def kernel(self, kernel: Union[Kernel, List[Kernel]]):
        ScoreTest.kernel.fset(self, kernel)
        if self.yidx is not None:
            x, y = tf.meshgrid(self.yidx, self.yidx)
            idx = tf.reshape(tf.stack((y, x), axis=2), (-1, 2))
            if tf.size(tf.shape(self._K)) > 2:
                bdim = tf.shape(self._K)[0]
                idx = tf.tile(idx, (bdim, 1))
                idx = tf.concat(
                    (
                        tf.repeat(
                            tf.range(bdim, dtype=self.yidx.dtype), tf.square(tf.size(self.yidx))
                        )[:, tf.newaxis],
                        idx,
                    ),
                    axis=1,
                )
            self._K = tf.squeeze(
                tf.reshape(
                    tf.gather_nd(self._K, idx), (-1, tf.size(self.yidx), tf.size(self.yidx)),
                )
            )

    def _fit_null(self, y: tf.Tensor) -> NullModel:
        scaledy = y / self.sizefactors
        res = minimize(
            self._negative_negbinom_loglik,
            x0=[
                tf.math.log(tf.reduce_mean(scaledy)),
                tf.math.log(
                    tf.maximum(1e-8, self._moments_dispersion_estimate(scaledy, self.sizefactors))
                ),
            ],
            args=(y, self.sizefactors),
            jac=self._grad_negative_negbinom_loglik,
            method="bfgs",
        )
        mu = tf.exp(res.x[0]) * self.sizefactors
        alpha = tf.exp(res.x[1])
        return self.NullModel(mu, alpha)

    def _test(
        self, y: tf.Tensor, nullmodel: NullModel
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        if self.yidx is not None:
            y = tf.gather(y, self.yidx)

        return self._do_test(
            tf.cast(y, self.dtype),
            tf.cast(nullmodel.alpha, self.dtype),
            tf.cast(nullmodel.mu, self.dtype),
        )

    @tf.function(experimental_compile=True)
    def _do_test(
        self, rawy: tf.Tensor, alpha: tf.Tensor, mu: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        W = mu / (1 + alpha * mu)
        stat = 0.5 * tf.reduce_sum(
            (rawy / mu - 1) * W * tf.tensordot(self._K, W * (rawy / mu - 1), axes=(-1, -1)), axis=-1
        )

        P = tf.linalg.diag(W) - W[:, tf.newaxis] * W[tf.newaxis, :] / tf.reduce_sum(W)
        PK = W[:, tf.newaxis] * self._K - W[:, tf.newaxis] * (
            (W[tf.newaxis, :] @ self._K) / tf.reduce_sum(W)
        )
        trace_PK = tf.linalg.trace(PK)
        e_tilde = 0.5 * trace_PK
        I_tau_tau = 0.5 * tf.reduce_sum(PK * PK, axis=(-2, -1))
        I_tau_theta = 0.5 * tf.reduce_sum(PK * P, axis=(-2, -1))
        I_theta_theta = 0.5 * tf.reduce_sum(tf.square(P), axis=(-2, -1))
        I_tau_tau_tilde = I_tau_tau - tf.square(I_tau_theta) / I_theta_theta

        return stat, e_tilde, I_tau_tau_tilde

    @staticmethod
    @tf.function(experimental_compile=True)
    def _moments_dispersion_estimate(y, sizefactors):
        """
        This is lifted from the first DESeq paper
        """
        v = tf.math.reduce_variance(y)
        m = tf.reduce_mean(y)
        s = tf.reduce_mean(1 / sizefactors)
        return (v - s * m) / tf.square(m)

    @staticmethod
    @tf.function(experimental_compile=True)
    def _negative_negbinom_loglik(params, counts, sizefactors):
        logmu = params[0]
        logalpha = params[1]
        mus = tf.exp(logmu) * sizefactors
        nexpalpha = tf.exp(-logalpha)
        ct_plus_alpha = counts + nexpalpha
        return -tf.reduce_sum(
            tf.math.lgamma(ct_plus_alpha)
            - tf.math.lgamma(nexpalpha)
            - ct_plus_alpha * tf.math.log(1 + tf.exp(logalpha) * mus)
            + counts * logalpha
            + counts * tf.math.log(mus)
            - tf.math.lgamma(counts + 1)
        )

    @staticmethod
    @tf.function(experimental_compile=True)
    def _grad_negative_negbinom_loglik(params, counts, sizefactors):
        logmu = params[0]
        logalpha = params[1]
        mu = tf.exp(logmu)
        mus = mu * sizefactors
        nexpalpha = tf.exp(-logalpha)
        one_alpha_mu = 1 + tf.exp(logalpha) * mus

        grad0 = tf.reduce_sum((counts - mus) / one_alpha_mu)  # d/d_mu
        grad1 = tf.reduce_sum(
            nexpalpha
            * (
                tf.math.digamma(nexpalpha)
                - tf.math.digamma(counts + nexpalpha)
                + tf.math.log(one_alpha_mu)
            )
            + (counts - mus) / one_alpha_mu
        )  # d/d_alpha
        return -tf.convert_to_tensor((grad0, grad1))
