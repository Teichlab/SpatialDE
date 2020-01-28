from time import time
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
import gpflow
import tensorflow as tf

from .util import SGPIPM, get_dtype

class GPModel(metaclass=ABCMeta):
    def __init__(self, model: gpflow.models.GPModel, optimizer):
        self.model = model
        self.optimizer = optimizer

        self._initial_vars = [v.value() for v in model.variables]
        self._gower = None
        self._Sigma = None

    @tf.function
    def _objective(self):
        return -self.model.log_marginal_likelihood()

    def optimize(self):
        self._s2_Sigma = None
        return self._optimize_impl()

    @abstractmethod
    def _optimize_impl(self):
        pass

    @staticmethod
    def _calc_gower(K):
        ''' Gower normalization factor for covariance matric K

        Based on https://github.com/PMBio/limix/blob/master/limix/utils/preprocess.py
        '''
        n = K.shape[0]
        P = tf.eye(n, dtype=K.dtype) - tf.ones((n, n), dtype=K.dtype) / n
        KP = K - tf.math.reduce_mean(K, 0, keepdims=True)
        trPKP = tf.linalg.trace(P * KP)

        return trPKP / (n - 1)

    def gower_scaling_factor(self):
        if self._gower is None:
            self._gower = self._gower_scaling_factor_impl()
        return self._gower

    @abstractmethod
    def _gower_scaling_factor_impl(self):
        pass

    def FSV(self):
        sg = self.model.kernel.variance.numpy() * self.gower_scaling_factor()
        return (sg / (sg + self.model.likelihood.variance)).numpy()

    @property
    def _s2_Sigma(self):
        if self._Sigma is None:
            with tf.GradientTape(persistent=True) as tape:
                y = self.model.log_marginal_likelihood()
                grads = tf.stack(tape.gradient(y, [self.model.kernel.variance, self.model.likelihood.variance]))
            obs_information = -tf.stack(tape.jacobian(grads, [self.model.kernel.variance, self.model.likelihood.variance], unconnected_gradients=tf.UnconnectedGradients.ZERO))
            self._Sigma = tf.linalg.inv(obs_information)
        return self._Sigma

    @_s2_Sigma.setter
    def _s2_Sigma(self, value=None):
        self._Sigma = value

    def s2_FSV(self):
        """
        Calculate the variance of the FSV estimate

        We use the negative of the Hessian of the margina log-likelihood as observed Fisher observation, the inverse of which is the
        asymptotic covariance matrix of the (s_s, s_e) estimate. We then use the Delta method to get the asymptotic variance of FSV.

        TODO: figure out how to handle the case of free inducing points
        """
        g = self.gower_scaling_factor()
        s_s = self.model.kernel.variance
        s_e = self.model.likelihood.variance
        nabla_FSV = tf.stack(([s_e * g / (s_s * g + s_e) ** 2], [-s_s * g / (s_s * g + s_e)**2]))
        return (tf.transpose(nabla_FSV) @ self._s2_Sigma @ nabla_FSV)[0,0].numpy()

    def s2_s_hat(self):
        return self._s2_Sigma[0,0].numpy()

    def s2_e_hat(self):
        return self._s2_Sigma[1,1].numpy()

    def reinitialize(self):
        for i, v in zip(self._initial_vars, self.model.variables):
            v.assign(i)
        self._s2_Sigma = None

class DeterministicGPModel(GPModel):
    def __init__(self, X, Y, model: gpflow.models.GPModel, optimizer, method="BFGS"):
        GPModel.__init__(self, model, optimizer)
        self.X = X
        self.Y = Y
        self._method = method

    def _optimize_impl(self):
        results = []
        for g in range(self.Y.shape[1]):
            self.reinitialize()
            self.model.data[1].assign(self.Y.iloc[:, g].to_numpy()[:, np.newaxis])
            self.model.likelihood.variance.assign(0.1) # apparently, this improves numerical stability during optimization

            t0 = time()
            self.optimizer.minimize(self._objective, variables=self.model.trainable_variables, method=self._method)
            t = time() - t0

            M = 0
            for v in self.model.trainable_variables:
                M += tf.math.reduce_prod(v.shape).numpy()

            res = {
                'g': self.Y.columns[g],
                'max_ll': -self._objective().numpy(),
                'max_mu_hat': self.model.mean_function.c[0].numpy(),
                'max_s2_s_hat': self.model.kernel.variance.numpy(),
                'max_s2_e_hat': self.model.likelihood.variance.numpy(),
                'time': t,
                'M': M,
                'n': self.Y.shape[0],
                'marginal_ll': self.model.log_marginal_likelihood().numpy(),
                'FSV': self.FSV(),
                's2_FSV': self.s2_FSV(),
                's2_s_hat': self.s2_s_hat(),
                's2_e_hat': self.s2_e_hat()
            }
            if hasattr(self.model.kernel, "lengthscale"):
                res['l'] = self.model.kernel.lengthscale.numpy()
            results.append(res)
        return pd.DataFrame(results)

    def _gower_scaling_factor_impl(self):
        K = self.model.kernel.K(self.X.to_numpy())
        return self._calc_gower(K)

class GPRModel(DeterministicGPModel):
    def __init__(self, X, Y, kernel, optimizer=gpflow.optimizers.Scipy()):
        m = gpflow.models.GPR(data=(X.to_numpy(), tf.Variable(initial_value=tf.zeros((Y.shape[0], 1), dtype=get_dtype(Y)), trainable=False)), kernel=kernel, mean_function=gpflow.mean_functions.Constant())
        DeterministicGPModel.__init__(self, X, Y, m, optimizer)

class SGPRModel(DeterministicGPModel):
    def __init__(self, X, Y, kernel, optimizer=gpflow.optimizers.Scipy(), rng: np.random.Generator = np.random.default_rng(), ipm: SGPIPM = SGPIPM.random, ninducers=None):
        if ninducers is None:
            ninducers = max(100, np.sqrt(X.shape[0]))
        if ipm == SGPIPM.free or ipm == SGPIPM.random:
            inducers = X.iloc[rng.integers(0, X.shape[0], ninducers), :].to_numpy()
        elif ipm == SGPIPM.grid:
            rngmin = X.min(0)
            rngmax = X.max(0)
            xvals = np.linspace(rngmin[0], rngmax[0], int(np.ceil(np.sqrt(ninducers))))
            yvals = np.linspace(rngmin[1], rngmax[1], int(np.ceil(np.sqrt(ninducers))))
            xx, xy = np.meshgrid(xvals, yvals)
            inducers = np.hstack((xx.reshape((xx.size, 1)), xy.reshape((xy.size, 1))))

        method = "BFGS"
        if ipm == SGPIPM.free and ninducers > 1e3:
            method = "L-BFGS-B"
            if hasattr(kernel, "lengthscale"):
                kernel.lengthscale.transform = gpflow.utilities.bijectors.positive(lower=0.5 * tf.math.reduce_min(gpflow.utilities.ops.square_distance(X.to_numpy())))
        m = gpflow.models.SGPR(data=(X.to_numpy(), tf.Variable(initial_value=tf.zeros((Y.shape[0], 1), dtype=get_dtype(Y)), trainable=False)), kernel=kernel, inducing_variable = inducers, mean_function=gpflow.mean_functions.Constant())

        DeterministicGPModel.__init__(self, X, Y, m, optimizer, method)
        self._rng = rng
        self._ipm = ipm
        self._ninducers = ninducers

    def _optimize_impl(self):
        if self._ipm == SGPIPM.free:
            self._gower = None
        return DeterministicGPModel.optimize()

    def _gower_scaling_factor_impl(self):
        K = self.model.kernel.K(self.model.inducing_variable.Z)
        return self._calc_gower(K)
