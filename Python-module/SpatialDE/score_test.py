from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import chi2
from scipy.special import loggamma, digamma
from scipy.optimize import minimize

from .models import TestableModel


@dataclass(frozen=True)
class ScoreTestResults:
    kappa: float
    nu: float
    U_tilde: float
    pval: float

    def to_dict(self):
        return {
            "pval": self.pval,
            "kappa": self.kappa,
            "U_tilde": self.U_tilde,
            "nu": self.nu,
        }


class ScoreTest(metaclass=ABCMeta):
    def __init__(
        self, X: np.ndarray, Y: np.ndarray, rawY: np.ndarray, model: Optional[TestableModel] = None
    ):
        self.model = model
        self._K_ = None

    @property
    def _K(self):
        if self._K_ is None:
            return self._getK()
        else:
            return self._K_

    def __enter__(self):
        self._K_ = self._getK()
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

    def _getK(self):
        if self.model is None:
            raise RuntimeError(
                "This ScoreTest does not have a model. Please assign a TestableModel to the model attribute of this ScoreTest object."
            )
        return self.model.K


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
    def __init__(
        self, X: np.ndarray, Y: np.ndarray, rawY: np.ndarray, model: Optional[TestableModel] = None
    ):
        super().__init__(X, Y, rawY, model)
        self.sizefactors = np.sum(rawY, axis=1) * 1e-3
        yidx = self.sizefactors.nonzero()[0]
        self.yidx = None
        if yidx.shape[0] != self.sizefactors.shape[0]:
            self.yidx = yidx
            self.sizefactors = np.take(self.sizefactors, self.yidx)

    def _getK(self):
        K = super()._getK()
        if self.yidx is not None:
            rawy = np.take(self.model.rawy, self.yidx)
            K = np.take(K, np.ravel_multi_index(np.ix_(self.yidx, self.yidx), K.shape))
        return K

    def __call__(self):
        rawy = self.model.rawy
        if self.yidx is not None:
            rawy = rawy[self.yidx]
        K = self._K

        scaledy = rawy / self.sizefactors
        res = minimize(
            self._negative_negbinom_loglik,
            x0=[np.log(np.mean(scaledy)), 0],
            args=(rawy, self.sizefactors),
            jac=self._grad_negative_negbinom_loglik,
            method="bfgs",
        )
        alpha = np.exp(res.x[1])
        mu = np.exp(res.x[0]) * self.sizefactors

        W = mu / (1 + alpha * mu)

        stat = 0.5 * np.sum((rawy / mu - 1) * W * np.dot(K, W * (rawy / mu - 1)))

        P = np.diag(W) - W[:, np.newaxis] * W[np.newaxis, :] / np.sum(W)
        PK = W[:, np.newaxis] * K - W[:, np.newaxis] * ((W[np.newaxis, :] @ K) / np.sum(W))
        trace_PK = np.trace(PK)
        e_tilde = 0.5 * trace_PK
        I_tau_tau = 0.5 * np.sum(PK * PK)
        I_tau_theta = 0.5 * np.sum(PK * P)
        I_theta_theta = 0.5 * np.sum(np.square(P))
        I_tau_tau_tilde = I_tau_tau - np.square(I_tau_theta) / I_theta_theta

        return self._calc_test(stat, e_tilde, I_tau_tau_tilde)

    def _moments_dispersion_estimate(self, y=None):
        """
        This is lifted from the first DESeq paper
        """
        if y is None:
            y = self.model.rawy / self.sizefactors
        v = np.var(y)
        m = np.mean(y)
        s = np.mean(1 / self.sizefactors)
        return (v - s * m) / np.square(m)

    @staticmethod
    def _negative_negbinom_loglik(params, counts, sizefactors):
        logmu = params[0]
        logalpha = params[1]
        mus = np.exp(logmu) * sizefactors
        nexpalpha = np.exp(-logalpha)
        ct_plus_alpha = counts + nexpalpha
        return -np.sum(
            loggamma(ct_plus_alpha)
            - loggamma(nexpalpha)
            - ct_plus_alpha * np.log(1 + np.exp(logalpha) * mus)
            + counts * logalpha
            + counts * np.log(mus)
            - loggamma(counts + 1)
        )

    @staticmethod
    def _grad_negative_negbinom_loglik(params, counts, sizefactors):
        logmu = params[0]
        logalpha = params[1]
        mu = np.exp(logmu)
        mus = mu * sizefactors
        nexpalpha = np.exp(-logalpha)
        one_alpha_mu = 1 + np.exp(logalpha) * mus
        grad = np.empty((2,))
        grad[0] = np.sum((counts - mus) / one_alpha_mu)  # d/d_mu
        grad[1] = np.sum(
            nexpalpha * (digamma(nexpalpha) - digamma(counts + nexpalpha) + np.log(one_alpha_mu))
            + (counts - mus) / one_alpha_mu
        )  # d/d_alpha
        return -grad
