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
    def __init__(self, model:Model):
        self.model = model

    @abstractmethod
    def __call__(self) -> ScoreTestResults:
        pass

class GaussianScoreTest(ScoreTest):
    def __call__(self):
        null_prediction = self._null_prediction()
        null_variance = self._null_variance()

        scaling = 1 / (2 * null_variance ** 2)
        PK = self.model.K - np.mean(self.model.K, axis=0, keepdims=True)

        I_tau_tau = scaling * np.sum(PK ** 2)
        I_tau_theta = scaling * np.trace(PK)  # P is idempotent
        I_theta_theta = scaling * self.model.n
        I_tau_tau_tilde = I_tau_tau - I_tau_theta ** 2 / I_theta_theta

        e_tilde = 1 / (2 * null_variance) * np.trace(PK)
        kappa = I_tau_tau / (2 * e_tilde)
        nu = 2 * e_tilde ** 2 / I_tau_tau

        res = self.model.y - null_prediction
        stat = scaling * np.dot(res, np.dot(self.model.K, res))

        ch2 = chi2(nu)
        pval = ch2.sf(stat / kappa)

        return ScoreTestResults(kappa, nu, stat, pval)

    @abstractmethod
    def _null_prediction(self):
        pass

    @abstractmethod
    def _null_variance(self):
        pass

class GaussianConstantScoreTest(GaussianScoreTest):
    def _null_prediction(self):
        return self.model.y.mean()

    def _null_variance(self):
        return self.model.y.var(ddof=0)

class GaussianNullScoreTest(GaussianScoreTest):
    def _null_prediction(self):
        return 0

    def _null_variance(self):
        return np.sum(np.square(self.model.y)) / self.model.n
