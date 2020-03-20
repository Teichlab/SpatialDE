from time import time
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
import gpflow
import tensorflow as tf

from . import util
from .util import SGPIPM, GP, GPControl, get_dtype
from .sm_kernel import *


class SMPlusLinearKernel(Sum):
    def __init__(self, sm_kernel):
        super().__init__([sm_kernel, gpflow.kernels.Linear()])

    @property
    def spectral_mixture(self):
        return self.kernels[0]

    @property
    def linear(self):
        return self.kernels[1]

    @staticmethod
    def _scaled_var(X, k):
        """ Gower normalization factor for covariance matric K

        Based on https://github.com/PMBio/limix/blob/master/limix/utils/preprocess.py
        """
        K = k(X)
        gower = (tf.linalg.trace(K) - tf.reduce_sum(tf.reduce_mean(K, axis=0))) / (
            K.shape[0] - 1
        )
        return k.variance * gower

    def scaled_variance(self, X):
        smvars = tf.convert_to_tensor([self._scaled_var(X, k) for k in self.kernels[0]])
        linvar = self._scaled_var(X, self.kernels[1])
        return tf.concat((smvars, tf.expand_dims(linvar, axis=0)), axis=0)

class GeneGPModel(metaclass=ABCMeta):
    @abstractmethod
    def freeze(self):
        pass

    @staticmethod
    def mixture_kernel(X, ncomponents=5, ard=True, minvar=1e-3):
        range = tf.reduce_min(tf.reduce_max(X, axis=0) - tf.reduce_min(X, axis=0))
        dist = tf.sqrt(gpflow.utilities.ops.square_distance(X, None))
        dist = tf.linalg.set_diag(
            dist, tf.fill((X.shape[0],), tf.cast(np.inf, dist.dtype))
        )
        min_1nndist = tf.reduce_min(dist)

        minperiod = 2 * min_1nndist
        varinit = min_1nndist + 0.5 * (range - min_1nndist)
        periodinit = minperiod + 0.5 * (range - minperiod)

        if ard:
            varinit = tf.repeat(varinit, X.shape[1])
            periodinit = tf.repeat(periodinit, X.shape[1])

        kernels = []
        for v in np.linspace(minvar, 1, ncomponents):
            k = Spectral(variance=v, lengthscale=varinit, period=periodinit)
            k.period.transform = gpflow.utilities.positive(lower=minperiod)
            kernels.append(k)
        kern = SpectralMixture(kernels)
        return SMPlusLinearKernel(kern)

    # this is a dirty hack to fix pickling of frozen models, see https://github.com/GPflow/GPflow/pull/1338
    @classmethod
    def _fix_frozen(cls, m):
        try:
            mvars = vars(m)
            for name in list(vars(m).keys()):
                var = mvars[name]
                if isinstance(var, tf.Tensor):
                    delattr(m, name)
                    setattr(m, name, var)
                elif isinstance(var, tf.Module):
                    cls._fix_frozen(var)
                elif isinstance(var, list):
                    for v in var:
                        cls._fix_frozen(var)
                elif isinstance(var, dict):
                    for v in var.values():
                        cls._fix_frozen(v)
        except TypeError:
            pass
        return m

class GPR(gpflow.models.GPR, GeneGPModel):
    def __init__(self, X, Y, n_kernel_components=5, ard=True, minvar=1e-3, dimnames=None):
        kern = self.mixture_kernel(X, n_kernel_components, ard, minvar)
        super().__init__(
            data=[X, Y], kernel=kern, mean_function=gpflow.mean_functions.Constant()
        )

    def freeze(self):
        X = self.data[0]
        self.data[0] = None
        frozen = gpflow.utilities.freeze(self)
        frozen.data[0] = self.data[0] = X
        return self._fix_frozen(frozen)

class SGPR(gpflow.models.SGPR, GeneGPModel):
    def __init__(
        self, X, Y, inducing_variable, n_kernel_components=5, ard=True, minvar=1e-3, dimnames=None
    ):
        kern = self.mixture_kernel(X, n_kernel_components, ard, minvar)
        super().__init__(
            data=[X, Y],
            kernel=kern,
            inducing_variable=inducing_variable,
            mean_function=gpflow.mean_functions.Constant(),
        )

    def freeze(self):
        X = self.data[0]
        self.data[0] = None

        trainable_inducers = self.inducing_variable.Z.trainable
        if not trainable_inducers:
            Z = self.inducing_variable.Z
        frozen = gpflow.utilities.freeze(self)
        frozen.data[0] = self.data[0] = X
        if not trainable_inducers:
            frozen.inducing_variable.Z = Z
        return self._fix_frozen(frozen)

@dataclass(frozen=True)
class VarPart:
    spectral_mixture: tf.Tensor
    linear: tf.Tensor
    noise: tf.Tensor

@dataclass(frozen=True)
class Variance:
    scaled_variance: VarPart
    fraction_variance: VarPart
    var_fraction_variance: VarPart

@dataclass(frozen=True)
class ScoreTest:
    kappa: util.float
    nu: util.float
    U_tilde: util.float
    pval: util.float

class GeneGP:
    def __init__(self, model: GeneGPModel, minimize_fun, *args, **kwargs):
        self.model = model

        self._frozen = False
        self._variancevars = list(self.model.kernel.parameters)
        self._variancevars.append(self.model.likelihood.variance)

        self._trainable_variance_idx = []
        offset = 0
        variancevars = set([v.experimental_ref() for v in self._variancevars])
        for v in self.model.parameters:
            if v.experimental_ref() in variancevars:
                self._trainable_variance_idx.extend(
                    [offset + int(i) for i in range(tf.size(v))]
                )
                offset += int(tf.size(v))

        self.__invHess = None
        self.model.likelihood.variance.assign(tf.math.reduce_variance(self.model.data[1]))
        self._optimize(minimize_fun, *args, **kwargs)
        self._freeze()

        self.score_test_results = self._score_test()

    @property
    def pval(self):
        return self.score_test_results.pval

    @property
    def kernel(self):
        return self.model.kernel

    def predict_mean(self, X=None):
        if X is None:
            X = self.model.data[0]
        return self.model.predict_f(X)[0]

    def plot_power_spectrum(self):
        pass

    @staticmethod
    def _concat_tensors(tens):
        return tf.concat([tf.reshape(t, (-1,)) for t in tens], axis=0)

    @property
    def _invHess(self):
        if self.__invHess is None:
            # tf.hessians doesn't work yet (https://github.com/tensorflow/tensorflow/issues/29781)
            # and tape.jacobian() doesn't like lengthscale and period parameters for some reason
            # (it aborts with AttributeError: Tensor.graph is meaningless when eager execution is enabled.).
            # So we need to do this the hard way
            with tf.GradientTape(persistent=True) as tape:
                y = self.model.log_marginal_likelihood()
                tape.watch(y)
                grad = self._concat_tensors(
                    tape.gradient(y, self.model.trainable_variables)
                )
                grads = tf.split(
                    grad, tf.ones((tf.size(grad),), dtype=tf.int32)
                )  # this is necessary to be able to get the gradient of each entry
            hess = tf.stack(
                [
                    self._concat_tensors(
                        tape.gradient(
                            g,
                            self.model.trainable_variables,
                            unconnected_gradients=tf.UnconnectedGradients.ZERO,
                        )
                    )
                    for g in grads
                ]
            )
            self._invHess = tf.linalg.inv(hess)
        return self.__invHess

    @_invHess.setter
    def _invHess(self, invhess):
        x, y = tf.meshgrid(self._trainable_variance_idx, self._trainable_variance_idx)
        invhess = tf.reshape(
            tf.gather_nd(
                invhess, tf.stack([tf.reshape(x, (-1,)), tf.reshape(y, (-1,))], axis=1)
            ),
            (len(self._trainable_variance_idx), len(self._trainable_variance_idx)),
        )
        self.__invHess = invhess

    def _optimize(self, minimize_fun, *args, **kwargs):
        res = minimize_fun(
            lambda: -self.model.log_marginal_likelihood(),
            self.model.trainable_variables,
            *args,
            **kwargs,
        )
        if isinstance(res, dict) and "hess_inv" in res:
            self._invHess = tf.cast(res["hess_inv"], gpflow.config.default_float())
        elif hasattr(res, "hess_inv"):
            self._invHess = tf.cast(res.hess_inv, gpflow.config.default_float())

    def _freeze(self):
        if self._frozen:
            return
        # this code calculates the variance of the fraction of spatial variance estimate
        # We use the negative of the Hessian of the marginal log-likelihood as observed Fisher observation, the inverse of which is the
        # asymptotic covariance matrix of the estimate. We then use the Delta method to get the asymptotic variance of FSV.
        # TODO: I'm not quite sure if this is valid for the case of free inducing points, since these are variational parameters
        with tf.GradientTape() as t:
            variances = self._concat_tensors(
                [
                    self.model.kernel.scaled_variance(self.model.data[0]),
                    tf.expand_dims(self.model.likelihood.variance, axis=0),
                ]
            )
            totalvar = tf.reduce_sum(variances)
            variance_fractions = variances / totalvar

        grads = t.jacobian(variance_fractions, self._variancevars)
        grads = tf.concat(
            [tf.expand_dims(g, -1) if g.ndim < 2 else g for g in grads], axis=1
        )
        errors = tf.reduce_sum((grads @ self._invHess) * grads, axis=1)

        variances = VarPart(variances[0:-2], variances[-2], variances[-1])
        variance_fractions = VarPart(
            variance_fractions[0:-2], variance_fractions[-2], variance_fractions[-1]
        )
        errors = VarPart(errors[0:-2], errors[-2], errors[-1])

        self.variances = Variance(variances, variance_fractions, errors)

        self.model = self.model.freeze()
        self._frozen = True

    def _score_test(self):
        mu = tf.reduce_mean(self.model.data[1])
        s2 = tf.math.reduce_variance(
            self.model.data[1]
        )  # Tensorflow calculates the biased MLE
        scaling = 1 / (2 * s2 ** 2)

        N = self.model.data[1].shape[0]
        P = tf.eye(N, dtype=mu.dtype) - tf.fill((N, N), tf.cast(1 / N, mu.dtype))
        K = self.model.kernel(self.model.data[0])
        PK = K - tf.reduce_mean(K, axis=0, keepdims=True)
        I_tau_tau = scaling * tf.reduce_sum(PK ** 2)
        I_tau_theta = scaling * tf.linalg.trace(PK)  # P is idempotent
        I_theta_theta = scaling * tf.linalg.trace(P)
        I_tau_tau_tilde = I_tau_tau - I_tau_theta ** 2 / I_theta_theta

        e_tilde = 1 / (2 * s2) * tf.linalg.trace(PK)
        kappa = I_tau_tau / (2 * e_tilde)
        nu = 2 * e_tilde ** 2 / I_tau_tau

        res = self.model.data[1] - mu
        stat = scaling * tf.squeeze(
            tf.tensordot(res, tf.tensordot(K, res, axes=[1, 0]), axes=[0, 0])
        )

        chi2 = tfp.distributions.Chi2(nu)
        pval = chi2.survival_function(stat / kappa)

        return ScoreTest(kappa.numpy(), nu.numpy(), stat.numpy(), pval.numpy())

class DataSetResults(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value: GeneGP):
        if not isinstance(value, GeneGP):
            raise TypeError("value must be a GeneGP object")
        super().__setitem__(key, value)

    def to_df(self):
        df = defaultdict(lambda: [])
        for gene, res in self.items():
            df["gene"].append(gene)
            df["pval"].append(res.pval)
            variances = res.variances
            df["FSV"].append(
                (
                    tf.reduce_sum(variances.fraction_variance.spectral_mixture)
                    + variances.fraction_variance.linear
                ).numpy()
            )
            df["s2_FSV"].append(variances.var_fraction_variance.noise)

            for i, k in enumerate(res.kernel.spectral_mixture.kernels):
                df["sm_variance_%i" % i] = k.variance
                df["sm_lengthscale_%i" % i] = k.lengthscale
                df["sm_period_%i" % i] = k.period
            df["linear_variance"] = res.kernel.linear.variance
            df["model"] = res
        df = pd.DataFrame(df)
        df["p.adj"] = util.bh_adjust(df.pval.to_numpy())
        return df.set_index('gene')
