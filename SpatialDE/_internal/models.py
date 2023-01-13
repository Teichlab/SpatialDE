from abc import ABCMeta, abstractmethod
from typing import Union
from dataclasses import dataclass

import numpy as np
import scipy
from scipy import optimize
from scipy.misc import derivative
from scipy.stats import chi2

from .kernels import Kernel


class Model:
    def __init__(self, X: np.ndarray, kernel: Kernel):
        self.X = X
        self.n = X.shape[0]
        self.kernel = kernel

        self._K = None
        self._y = None

    def _reset(self):
        pass

    @property
    def K(self):
        if self._K is not None:
            return self._K
        else:
            return self.kernel.K(self.X)

    @property
    def n_parameters(self) -> float:
        return 0

    @property
    def FSV(self) -> float:
        return np.nan

    @property
    def s2_FSV(self) -> float:
        return np.nan

    @property
    def logdelta(self) -> float:
        return np.nan

    @logdelta.setter
    def logdelta(self, ld: float):
        pass

    @property
    def s2_logdelta(self) -> float:
        return np.nan

    @property
    def delta(self) -> float:
        return np.nan

    @delta.setter
    def delta(self, nd: float):
        pass

    @property
    def s2_delta(self) -> float:
        return np.nan

    @property
    def mu(self) -> float:
        return np.nan

    @property
    def sigma_s2(self) -> float:
        return np.nan

    @property
    def sigma_n2(self) -> float:
        return np.nan

    @property
    def y(self) -> np.ndarray:
        return self._y

    @y.setter
    def y(self, newy: np.ndarray):
        self._y = newy
        self._reset()
        self._ychanged()

    def optimize(self):
        pass

    @property
    def log_marginal_likelihood(self) -> float:
        self._check_y()
        return self._lml()

    def _lml(self):
        return np.nan

    def _ychanged(self):
        pass

    def _check_y(self):
        if self.y is None:
            raise RuntimeError("assign observed values first")
        if self.y.shape[0] != self.n:
            raise RuntimeError("different numbers of observations in y and X")

    def __enter__(self):
        self._K = self.K

    def __exit__(self, *args):
        self._K = None


class GPModel(Model):
    def __init__(self, X: np.ndarray, kernel: Kernel):
        super().__init__(X, kernel)

        self._y = None
        self._logdelta = 0
        self._reset()

    def __enter__(self):
        super().__enter__()
        self._reset()
        # Gower normalization factor for covariance matric K
        # Based on https://github.com/PMBio/limix/blob/master/limix/utils/preprocess.py
        self.gower = (np.trace(self.K) - np.sum(np.mean(self.K, axis=0))) / (self.n - 1)

        return self

    def __exit__(self, *args):
        super().__exit__(*args)

    def _reset(self):
        self._s2_FSV = None
        self._s2_delta = None
        self._s2_logdelta = None
        self._mu = None
        self._sigma_s2 = None
        self._sigma_n2 = None

    @property
    @abstractmethod
    def n_parameters(self):
        pass

    @property
    def FSV(self):
        return self.gower / (self.delta + self.gower)

    @property
    def s2_FSV(self):
        if self._s2_FSV is None:
            self._s2_FSV = self._calc_s2_FSV()
        return self._s2_FSV

    @property
    def logdelta(self) -> float:
        return self._logdelta

    @logdelta.setter
    def logdelta(self, ld: float):
        self._logdelta = ld
        self._reset()
        self._logdeltachanged()

    @property
    def s2_logdelta(self) -> float:
        if self._s2_logdelta is None:
            self._s2_logdelta = self._calc_s2_logdelta()
        return self._s2_logdelta

    @property
    def s2_delta(self) -> float:
        if self._s2_delta is None:
            self._s2_delta = self._calc_s2_delta()
        return self._s2_delta

    def _logdeltachanged(self):
        pass

    @property
    def delta(self) -> float:
        return np.exp(self.logdelta)

    @delta.setter
    def delta(self, d: float):
        self.logdelta = np.log(d)

    def _objective(self, func):
        def obj(logdelta):
            self.logdelta = logdelta
            return func()

        return obj

    def optimize(self):
        res = optimize.minimize(
            self._objective(lambda: -self.log_marginal_likelihood),
            10,
            method="l-bfgs-b",
            bounds=[(-10, 20)],
            jac=False,
            options={"eps": 1e-4},
        )
        self.logdelta = res.x[0]
        return res

    @abstractmethod
    def _lml(self):
        pass

    @property
    def mu(self) -> float:
        if self._mu is None:
            self._mu = self._calc_mu()
        return self._mu

    @property
    def sigma_s2(self) -> float:
        if self._sigma_s2 is None:
            self._sigma_s2 = self._calc_sigma_s2()
        return self._sigma_s2

    @property
    def sigma_n2(self) -> float:
        if self._sigma_n2 is None:
            self._sigma_n2 = self._calc_sigma_n2()
        return self._sigma_n2

    @abstractmethod
    def _calc_mu(self) -> float:
        pass

    @abstractmethod
    def _calc_sigma_s2(self) -> float:
        pass

    @abstractmethod
    def _calc_sigma_s2(self) -> float:
        pass

    def _calc_s2_logdelta(self) -> float:
        ld = self.logdelta
        s2 = -1 / derivative(
            self._objective(lambda: self.log_marginal_likelihood), self.logdelta, n=2
        )
        self.logdelta = ld
        return s2

    def _calc_s2_delta(self) -> float:
        return self.s2_logdelta * self.delta

    def _calc_s2_FSV(self) -> float:
        ld = self.logdelta
        der = derivative(self._objective(lambda: self.FSV), self.logdelta, n=1)
        self.logdelta = ld
        return der**2 / self.s2_logdelta


class SGPR(GPModel):
    def __init__(self, X: np.ndarray, Z: np.ndarray, kern: Kernel):
        super().__init__(X, kernel=kern)
        self.Z = Z
        self.z = Z.shape[0]

    def __enter__(self):
        super().__enter__()
        K_uu = self.kernel.K(self.Z)
        K_uf = self.kernel.K(self.Z, self.X)
        K_ff = self.kernel.K_diag(self.X)

        L = np.linalg.cholesky(K_uu + 1e-6 * np.eye(self.z))
        LK_uf = scipy.linalg.solve_triangular(L, K_uf, lower=True)
        A = LK_uf @ LK_uf.T

        self._Lambda, U = np.linalg.eigh(A)
        self._B = U.T @ LK_uf
        self._B1 = np.sum(self._B, axis=-1)
        self._By = None
        self._traceterm = np.sum(K_ff) - np.trace(A)

        return self

    def __exit__(self, *args):
        super().__exit__(*args)
        self._Lambda = None
        self._B = None
        self._B1 = None
        self._By = None
        self._traceterm = None

    @property
    def n_parameters(self) -> float:
        return 3

    def _lml(self) -> float:
        delta = self.delta
        return 0.5 * (
            -self.n * np.log(2 * np.pi)
            - self.z * np.log(self.sigma_s2)
            - (self.n - self.z) * np.log(self.sigma_n2)
            - np.sum(np.log(delta + self._Lambda))
            - self.n
            - self._traceterm / delta
        )

    def _residual_quadratic(self) -> float:
        self._check_y()
        return (
            np.sum(self.y**2)
            - 2 * np.sum(self.y) * self.mu
            + self.mu**2 * self.n
            - np.sum((self._By - self._B1 * self.mu) ** 2 / self._dL)
        )

    def _calc_mu(self):
        self._check_y()
        sy = np.sum(self.y)
        ytones = np.sum(self._By * self._B1 / self._dL)
        onesones = np.sum(self._B1**2 / self._dL)

        return (sy - ytones) / (self.n - onesones)

    def _calc_sigma_s2(self):
        return self.sigma_n2 / self.delta

    def _calc_sigma_n2(self):
        return self._residual_quadratic() / self.n

    def _ychanged(self):
        if self._B is not None:
            self._By = np.dot(self._B, self.y)

    def _logdeltachanged(self):
        self._dL = self.delta + self._Lambda


class GPR(GPModel):
    def __init__(self, X: np.ndarray, kern: Kernel):
        super().__init__(X, kernel=kern)

    def __enter__(self):
        super().__enter__()

        K = self.kernel.K(self.X)
        self._Lambda, self._U = np.linalg.eigh(K)
        self._U1 = np.sum(self._U, axis=0)
        self._Uy = None

        return self

    def __exit__(self, *args):
        super().__exit__(*args)
        self._Lambda = self._U = None
        self._U1 = None
        self._Uy = None

    @property
    def n_parameters(self) -> float:
        return 3

    def _lml(self) -> float:
        return 0.5 * (
            -self.n * np.log(2 * np.pi)
            - np.sum(np.log(self._dL))
            - self.n
            - self.n * np.log(self.sigma_s2)
        )

    def _residual_quadratic(self) -> float:
        return np.sum((self._Uy - self._U1 * self.mu) ** 2 / self._dL)

    def _calc_mu(self) -> float:
        return np.sum(self._U1 * self._Uy / self._dL) / np.sum(np.square(self._U1) / self._dL)

    def _calc_sigma_s2(self) -> float:
        return self._residual_quadratic() / self.n

    def _calc_sigma_n2(self) -> float:
        return self.sigma_s2 * self.delta

    def _ychanged(self):
        if self._U is not None:
            self._Uy = np.dot(self._U.T, self.y)

    def _logdeltachanged(self):
        self._dL = self.delta + self._Lambda


class Constant(Model):
    def __init__(self, X: np.ndarray):
        super().__init__(X, Kernel())

    def _reset(self):
        self._mu = None
        self._s2 = None

    @property
    def n_parameters(self) -> float:
        return 2

    def optimize(self):
        self._check_y()
        self._mu = np.mean(self.y)
        self._s2 = np.var(self.y, ddof=0)
        return optimize.OptimizeResult(success=True)

    @property
    def mu(self) -> float:
        if self._mu is None:
            self.optimize()
        return self._mu

    @property
    def sigma_n2(self) -> float:
        if self._s2 is None:
            self.optimize()
        return self._s2

    def _lml(self) -> float:
        return (
            -0.5 * self.n * np.log(2 * np.pi * self.sigma_n2)
            - 0.5 * np.sum(np.square(self.y - self.mu)) / self.sigma_n2
        )


class Null(Model):
    def __init__(self, X: np.ndarray):
        super().__init__(X, Kernel())

    def _reset(self):
        self._s2 = None

    @property
    def n_parameters(self) -> float:
        return 1

    def optimize(self):
        self._check_y()
        self._s2 = np.sum(np.square(self.y)) / self.n
        return optimize.OptimizeResult(success=True)

    @property
    def sigma_n2(self) -> float:
        if self._s2 is None:
            self.optimize()
        return self._s2

    def _lml(self) -> float:
        return (
            -0.5 * self.n * np.log(2 * np.pi * self.sigma_n2)
            - 0.5 * np.square(self.y) / self.sigma_n2
        )


def model_factory(X: np.ndarray, Z: Union[np.ndarray, None], kern: Kernel, *args, **kwargs):
    if Z is None:
        return GPR(X, kern, *args, **kwargs)
    else:
        return SGPR(X, Z, kern, *args, **kwargs)
