from time import time
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
import gpflow
import tensorflow as tf

from .models import Model
from .sm_kernel import *
from .util import gower_factor


class Linear(gpflow.kernels.Linear):
    def K_novar(self, X, X2=None):
        if X2 is None:
            return tf.matmul(X, X, transpose_b=True)
        else:
            return tf.tensordot(X, X2, [-1, -1])


class SMPlusLinearKernel(gpflow.kernels.Sum):
    def __init__(self, sm_kernel):
        super().__init__([sm_kernel, Linear()])

    def K_novar(self, X, X2=None):
        return self._reduce([k.K_novar(X, X2) for k in self.kernels])

    @property
    def spectral_mixture(self):
        return self.kernels[0]

    @property
    def linear(self):
        return self.kernels[1]

    @staticmethod
    def _scaled_var(X, k):
        return gower_factor(k(X), k.variance)

    def scaled_variance(self, X):
        smvars = tf.convert_to_tensor([self._scaled_var(X, k) for k in self.kernels[0]])
        linvar = self._scaled_var(X, self.kernels[1])
        return tf.concat((smvars, tf.expand_dims(linvar, axis=0)), axis=0)


class GeneGPModel(metaclass=ABCMeta):
    @abstractmethod
    def freeze(self):
        pass

    @staticmethod
    def mixture_kernel(X, Y, ncomponents=5, ard=True, minvar=1e-3):
        range = tf.reduce_min(tf.reduce_max(X, axis=0) - tf.reduce_min(X, axis=0))
        dist = tf.sqrt(gpflow.utilities.ops.square_distance(X, None))
        dist = tf.linalg.set_diag(dist, tf.fill((X.shape[0],), tf.cast(np.inf, dist.dtype)))
        min_1nndist = tf.reduce_min(dist)

        datarange = tf.math.reduce_max(
            tf.math.reduce_max(X, axis=0) - tf.math.reduce_min(X, axis=0)
        )

        minperiod = 2 * min_1nndist
        varinit = min_1nndist + 0.5 * (range - min_1nndist)
        periodinit = minperiod + 0.5 * (range - minperiod)

        if ard:
            varinit = tf.repeat(varinit, X.shape[1])
            periodinit = tf.repeat(periodinit, X.shape[1])

        maxvar = 10 * tf.math.reduce_variance(Y)
        kernels = []
        for v in np.linspace(minvar, np.minimum(1, 0.9 * maxvar), ncomponents):
            k = Spectral(
                variance=gpflow.Parameter(
                    v,
                    transform=tfp.bijectors.Sigmoid(
                        low=gpflow.utilities.to_default_float(0),
                        high=gpflow.utilities.to_default_float(maxvar),
                    ),
                ),
                lengthscales=gpflow.Parameter(
                    varinit, transform=tfp.bijectors.Sigmoid(low=0.1 * min_1nndist, high=datarange)
                ),
                periods=gpflow.Parameter(
                    periodinit, transform=tfp.bijectors.Sigmoid(low=minperiod, high=2 * datarange)
                ),
            )
            kernels.append(k)
        kern = SpectralMixture(kernels)
        return SMPlusLinearKernel(kern)


class GPR(gpflow.models.GPR, GeneGPModel):
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_kernel_components: int = 5,
        ard: bool = True,
        minvar: float = 1e-3,
    ):
        kern = self.mixture_kernel(X, Y, n_kernel_components, ard, minvar)
        super().__init__(data=[X, Y], kernel=kern, mean_function=gpflow.mean_functions.Constant())

    def freeze(self):
        X = self.data[0]
        self.data = (None, self.data[1])
        frozen = gpflow.utilities.freeze(self)
        frozen.data = self.data = (X, self.data[1])
        return frozen


class SGPR(gpflow.models.SGPR, GeneGPModel):
    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        inducing_variable: Union[np.ndarray, gpflow.inducing_variables.InducingPoints],
        n_kernel_components: int = 5,
        ard: bool = True,
        minvar: float = 1e-3,
    ):
        kern = self.mixture_kernel(X, Y, n_kernel_components, ard, minvar)
        super().__init__(
            data=[X, Y],
            kernel=kern,
            inducing_variable=inducing_variable,
            mean_function=gpflow.mean_functions.Constant(),
        )

    def freeze(self):
        X = self.data[0]
        self.data = (None, self.data[1])

        trainable_inducers = self.inducing_variable.Z.trainable
        if not trainable_inducers:
            Z = self.inducing_variable.Z
        frozen = gpflow.utilities.freeze(self)
        frozen.data = self.data = (X, self.data[1])
        if not trainable_inducers:
            frozen.inducing_variable.Z = Z
        return frozen

    def log_marginal_likelihood(self):
        return self.elbo()


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


class GeneGP(Model):
    def __init__(self, model: GeneGPModel, minimize_fun, *args, **kwargs):
        self.model = model

        self._frozen = False
        self._variancevars = list(self.model.kernel.parameters)
        self._variancevars.append(self.model.likelihood.variance)

        self._trainable_variance_idx = []
        offset = 0
        variancevars = set(self._variancevars)
        for v in self.model.parameters:
            if v in variancevars:
                self._trainable_variance_idx.extend([offset + int(i) for i in range(tf.size(v))])
                offset += int(tf.size(v))

        self.__invHess = None
        self.model.likelihood.variance.assign(tf.math.reduce_variance(self.model.data[1]))

        t0 = time()
        self._optimize(minimize_fun, *args, **kwargs)
        self._freeze()
        t = time() - t0

        self._time = t

    @property
    def kernel(self):
        return self.model.kernel

    @property
    def K(self):
        return self.model.kernel.K_novar(self.model.data[0])

    @property
    def y(self):
        return tf.squeeze(self.model.data[1]).numpy()

    def predict_mean(self, X=None):
        if X is None:
            X = self.model.data[0]
        return self.model.predict_f(X)[0]

    def plot_power_spectrum(self, xlim: float = None, ylim: float = None, **kwargs):
        return self.model.kernel.spectral_mixture.plot_power_spectrum(xlim, ylim, **kwargs)

    @property
    def time(self):
        return self._time

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
                grad = self._concat_tensors(tape.gradient(y, self.model.trainable_variables))
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
            tf.gather_nd(invhess, tf.stack([tf.reshape(x, (-1,)), tf.reshape(y, (-1,))], axis=1)),
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
            self._invHess = gpflow.utilities.to_default_float(res["hess_inv"])
        elif hasattr(res, "hess_inv"):
            self._invHess = gpflow.utilities.to_default_float(res.hess_inv)

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

        grads = t.jacobian(
            variance_fractions, [v.unconstrained_variable for v in self._variancevars]
        )
        grads = tf.concat([tf.expand_dims(g, -1) if g.ndim < 2 else g for g in grads], axis=1)
        errors = tf.reduce_sum((grads @ self._invHess) * grads, axis=1)

        variances = VarPart(variances[0:-2], variances[-2], variances[-1])
        variance_fractions = VarPart(
            variance_fractions[0:-2], variance_fractions[-2], variance_fractions[-1]
        )
        errors = VarPart(errors[0:-2], errors[-2], errors[-1])

        self.variances = Variance(variances, variance_fractions, errors)

        self.model = self.model.freeze()
        self._frozen = True


class DataSetResults(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value: GeneGP):
        if not isinstance(value, GeneGP):
            raise TypeError("value must be a GeneGP object")
        super().__setitem__(key, value)

    def to_df(self, modelcol: str = "model"):
        df = defaultdict(lambda: [])
        for gene, res in self.items():
            df["gene"].append(gene)
            variances = res.variances
            df["FSV"].append((1 - variances.fraction_variance.noise).numpy())
            df["s2_FSV"].append(variances.var_fraction_variance.noise.numpy())

            for i, k in enumerate(res.kernel.spectral_mixture.kernels):
                df["sm_variance_%i" % i].append(k.variance.numpy())
                df["sm_lengthscale_%i" % i].append(k.lengthscales.numpy())
                df["sm_period_%i" % i].append(k.periods.numpy())
            df["linear_variance"].append(res.kernel.linear.variance.numpy())
            df["noise_variance"].append(res.model.likelihood.variance.numpy())
            df["time"].append(res.time)
            df[modelcol].append(res)
        df = pd.DataFrame(df)
        return df
