from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2

from .models import Model


@dataclass(frozen=True)
class ScoreTestResults:
    kappa: float
    nu: float
    U_tilde: float
    pval: float


class ScoreTest(metaclass=ABCMeta):
    def __init__(self, X: np.ndarray, Y: np.ndarray, rawY: np.ndarray, model: Model):
        self.model = model
        self._K_ = None

    @property
    def _K(self):
        if self._K_ is None:
            return self.model.K
        else:
            return self._K_

    def __enter__(self):
        self._K_ = self._K
        return self

    def __exit__(self, *args):
        self._K_ = None

    @abstractmethod
    def __call__(self) -> ScoreTestResults:
        pass

    @staticmethod
    def _calc_test(stat, e_tilde, I_tau_tau):
        kappa = I_tau_tau / (2 * e_tilde)
        nu = 2 * e_tilde ** 2 / I_tau_tau
        ch2 = chi2(nu)
        pval = ch2.sf(stat / kappa)
        return ScoreTestResults(kappa, nu, stat, pval)


class GaussianScoreTest(ScoreTest):
    pass


class GaussianConstantScoreTest(GaussianScoreTest):
    def __call__(self):
        null_prediction = self.model.y.mean()
        null_variance = self.model.y.var(ddof=0)

        scaling = 1 / (2 * null_variance ** 2)
        PK = self._K - np.mean(self._K, axis=0, keepdims=True)

        I_tau_tau = scaling * np.sum(PK ** 2)
        I_tau_theta = scaling * np.trace(PK)  # P is idempotent
        I_theta_theta = scaling * self.model.n
        I_tau_tau_tilde = I_tau_tau - I_tau_theta ** 2 / I_theta_theta

        e_tilde = 1 / (2 * null_variance) * np.trace(PK)

        res = self.model.y - null_prediction
        stat = scaling * np.sum(res * np.dot(self._K, res))
        return self._calc_test(stat, e_tilde, I_tau_tau_tilde)


class GaussianNullScoreTest(GaussianScoreTest):
    def __call__(self):
        null_variance = np.sum(np.square(self.model.y)) / self.model.n

        scaling = 1 / (2 * null - null_variance ** 2)
        I_tau_tau = scaling * np.sum(np.square(self._K))
        I_tau_theta = scaling * np.trace(self._K)
        I_theta_theta = scaling * self.model.n
        I_tau_tau_tilde = I_tau_tau - I_tau_theta ** 2 / I_theta_theta

        e_tilde = np.trace(self._K) / (2 * null_variance)

        stat = np.sum(self.model.y * np.dot(self._K, self.model.y))
        return self._calc_test(stat, e_tilde, I_tau_tau_tilde)


class NegativeBinomialScoreTest(ScoreTest):
    def __init__(self, X: np.ndarray, Y: np.ndarray, rawY: np.ndarray, model: Model):
        super().__init__(X, Y, rawY, model)
        self.sizefactors = np.sum(rawY, axis=1) / 1e3
        self.yidx = self.sizefactors.nonzero()[0]
        self.sizefactors = self.sizefactors[self.yidx]

    def __call__(self):
        rawy = self.model.rawy[self.yidx]
        K = self._K[np.ix_(self.yidx, self.yidx)]
        y = rawy / self.sizefactors
        alpha = self._moments_dispersion_estimate(y)
        if alpha < 0:
            alpha = 0  # we do not allow underdispersion. alpha=0 reduces to a Poisson null model

        mu = y.mean() * self.sizefactors
        delta = 1 / mu
        W = mu / (1 + alpha * mu)

        stat = 0.5 * np.sum((rawy / mu - 1) * W * np.dot(K, W * (rawy / mu - 1)))

        PK = W * K - W[:, np.newaxis] * W[np.newaxis, :] @ K / np.sum(W)
        trace_PK = np.trace(PK)
        e_tilde = 0.5 * trace_PK
        I_tau_tau_tilde = 0.5 * np.sum(
            PK * PK
        )  # I_tau_theta and I_theta_theta are zero since phi is fixed

        return self._calc_test(stat, e_tilde, I_tau_tau_tilde)

    def _moments_dispersion_estimate(self, y=None):
        """
        This is lifted from the first DESeq paper
        """
        if y is None:
            y = self.model.rawy / self.sizefactors
        v = y.var()
        m = y.mean()
        s = np.mean(1 / self.sizefactors)
        return (v - s * m) / np.square(m)
