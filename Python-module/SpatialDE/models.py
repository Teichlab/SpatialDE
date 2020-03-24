from abc import ABCMeta, abstractmethod
from typing import Union
from dataclasses import dataclass

import numpy as np
import scipy
from scipy import optimize
from scipy.misc import derivative
from scipy.stats import chi2

from .kernels import Kernel


@dataclass(frozen=True)
class ScoreTest:
    kappa: float
    nu: float
    U_tilde: float
    pval: float


class Model(metaclass=ABCMeta):
    def __init__(self, X: np.ndarray, kernel: Kernel):
        self.X = X
        self.n = X.shape[0]
        self.kernel = kernel
        self.K = self.kernel.K(self.X)

        self._y = None

    def _reset(self):
        pass

    def score_test(
        self, null_prediction: Union[float, np.ndarray], null_variance: float
    ) -> ScoreTest:
        scaling = 1 / (2 * null_variance ** 2)
        P = np.eye(self.n) - np.full((self.n, self.n), 1 / self.n)
        PK = self.K - np.mean(self.K, axis=0, keepdims=True)

        I_tau_tau = scaling * np.sum(PK ** 2)
        I_tau_theta = scaling * np.trace(PK)  # P is idempotent
        I_theta_theta = scaling * np.trace(P)
        I_tau_tau_tilde = I_tau_tau - I_tau_theta ** 2 / I_theta_theta

        e_tilde = 1 / (2 * null_variance) * np.trace(PK)
        kappa = I_tau_tau / (2 * e_tilde)
        nu = 2 * e_tilde ** 2 / I_tau_tau

        res = self.y - null_prediction
        stat = scaling * np.dot(res, np.dot(self.K, res))

        ch2 = chi2(nu)
        pval = ch2.sf(stat / kappa)

        return ScoreTest(kappa, nu, stat, pval)

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
    def y(self, ny: np.ndarray):
        self._y = ny
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


class GPModel(Model):
    def __init__(self, X: np.ndarray, kernel: Kernel):
        super().__init__(X, kernel)

        K = kernel.K(X)
        # Gower normalization factor for covariance matric K
        # Based on https://github.com/PMBio/limix/blob/master/limix/utils/preprocess.py
        self.gower = (np.trace(self.K) - np.sum(np.mean(self.K, axis=0))) / (self.n - 1)

        self._y = None
        self._logdelta = 0
        self._reset()

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
        return der ** 2 / self.s2_logdelta


class SGPR(GPModel):
    def __init__(self, X: np.ndarray, Z: np.ndarray, kern: Kernel):
        super().__init__(X, kernel=kern)

        K_uu = kern.K(Z)
        K_uf = kern.K(Z, X)
        K_ff = kern.K_diag(X)

        L = np.linalg.cholesky(K_uu + 1e-6 * np.eye(Z.shape[0]))
        LK_uf = scipy.linalg.solve_triangular(L, K_uf, lower=True)
        A = LK_uf @ LK_uf.T

        self._Lambda, U = np.linalg.eigh(A)
        self._B = U.T @ LK_uf
        self._B1 = np.sum(self._B, axis=-1)
        self._By = None
        self._traceterm = np.sum(K_ff) - np.sum(LK_uf ** 2)

    @property
    def n_parameters(self) -> float:
        return 3

    def _lml(self) -> float:
        delta = self.delta
        return 0.5 * (
            -self.n * np.log(2 * np.pi)
            - self.n * self.sigma_s2
            - np.sum(np.log(delta + self._Lambda))
            - self.n
            - self._traceterm / delta
        )

    def _residual_quadratic(self) -> float:
        self._check_y()
        return (
            np.sum(self.y ** 2)
            - 2 * np.sum(self.y) * self.mu
            + self.mu ** 2 * self.n
            - np.sum((self._By - self._B1 * self.mu) ** 2 / self._dL)
        )

    def _calc_mu(self):
        self._check_y()
        sy = np.sum(self.y)
        ytones = np.sum(self._By * self._B1 / self._dL)
        onesones = np.sum(self._B1 ** 2 / self._dL)

        return (sy - ytones) / (self.n - onesones)

    def _calc_sigma_s2(self):
        return self.sigma_n2 / self.delta

    def _calc_sigma_n2(self):
        return self._residual_quadratic() / self.n

    def _ychanged(self):
        self._By = np.dot(self._B, self.y)

    def _logdeltachanged(self):
        self._dL = self.delta + self._Lambda


class GPR(GPModel):
    def __init__(self, X: np.ndarray, kern: Kernel):
        super().__init__(X, kernel=kern)
        K = kern.K(X)
        self._Lambda, self._U = np.linalg.eigh(K)
        self._U1 = np.sum(self._U, axis=0)
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
        return np.sum(self._U1 * self._Uy / self._dL) / np.sum(
            np.square(self._U1) / self._dL
        )

    def _calc_sigma_s2(self) -> float:
        return self._residual_quadratic() / self.n

    def _calc_sigma_n2(self) -> float:
        return self.sigma_s2 * self.delta

    def _ychanged(self):
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


def model_factory(
    X: np.ndarray, Z: Union[np.ndarray, None], kern: Kernel, *args, **kwargs
):
    if Z is None:
        return GPR(X, kern, *args, **kwargs)
    else:
        return SGPR(X, Z, kern, *args, **kwargs)
