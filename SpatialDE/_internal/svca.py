from collections import namedtuple
from typing import Optional, List

import numpy as np
import scipy

import tensorflow as tf
import gpflow
from gpflow import default_float
from gpflow.utilities import to_default_float

from .util import gower_factor, quantile_normalize
from .score_test import ScoreTest
from .optimizer import MultiScipyOptimizer


class SVCA(tf.Module):
    _fracvar = namedtuple("FractionVariance", "intrinsic environmental noise")
    _fracvar_interact = namedtuple("FractionVraiance", "intrinsic environmental interaction noise")

    def __init__(
        self,
        expression: np.ndarray,
        X: np.ndarray,
        sizefactors: np.ndarray,
        kernel: Optional[gpflow.kernels.Kernel] = None,
    ):
        self.expression = to_default_float(expression)
        self.sizefactors = to_default_float(sizefactors)
        self._ncells, self._ngenes = tf.shape(self.expression)
        self._X = to_default_float(X)

        self._current_expression = tf.Variable(
            tf.zeros((self._ncells, self._ngenes - 1), dtype=default_float()), trainable=False
        )
        self.intrinsic_variance_matrix = tf.Variable(
            tf.zeros((self._ncells, self._ncells), dtype=default_float()), trainable=False
        )
        self._sigmas = gpflow.Parameter(
            tf.ones((4,), dtype=default_float()), transform=gpflow.utilities.positive(lower=1e-9)
        )
        self._currentgene = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.muhat = tf.Variable(tf.ones((self._ncells,), dtype=default_float()), trainable=False)

        self.kernel = kernel

        self._opt = MultiScipyOptimizer(lambda: -self.profile_log_reml(), self.trainable_variables)
        self._use_interactions = tf.Variable(False, dtype=tf.bool, trainable=False)

        self._old_interactions = False

    @property
    def sizefactors(self) -> np.ndarray:
        return self._sizefactors

    @sizefactors.setter
    def sizefactors(self, sizefactors: np.ndarray):
        self._sizefactors = np.squeeze(sizefactors)
        if len(self._sizefactors.shape) != 1:
            raise ValueError("Size factors vector must have rank 1")
        self._log_sizefactors = tf.squeeze(tf.math.log(to_default_float(sizefactors)))

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, kern):
        self._kernel = kern
        self._init_kern = gpflow.utilities.read_values(kern)

    @property
    def currentgene(self) -> int:
        return self._currentgene.numpy()

    @currentgene.setter
    def currentgene(self, gene: int):
        gene = tf.cast(gene, self._ngenes.dtype)
        if gene < 0 or gene >= self._ngenes:
            raise IndexError(f"gene must be between 0 and {self._ngenes}")

        self._currentgene.assign(gene)
        idx = [] if gene == 0 else tf.range(gene)
        idx = (
            tf.concat((idx, tf.range(gene + 1, self._ngenes)), axis=0)
            if gene < self._ngenes - 1
            else idx
        )
        self._current_expression.assign(
            tf.gather(self.expression, idx, axis=1) / self._sizefactors[:, tf.newaxis]
        )

        intvar = tf.matmul(self._current_expression, self._current_expression, transpose_b=True)
        self.intrinsic_variance_matrix.assign(intvar / gower_factor(intvar))

        muhat = self.expression[:, gene]
        muhat = tf.where(muhat < 1, 1, muhat)  # avoid problems with log link
        self.muhat.assign(muhat)

        self._sigmas.assign(
            tf.fill(
                (4,),
                0.25
                * tf.math.reduce_variance(
                    tf.math.log(self.expression[:, gene] + 1) - self._log_sizefactors
                ),
            )
        )
        if self._kernel is not None:
            gpflow.utilities.multiple_assign(self._kernel, self._init_kern)

    def profile_log_reml(self):
        Vchol = tf.linalg.cholesky(self.V())
        r = self._r(Vchol)
        quad = tf.tensordot(
            r,
            tf.squeeze(tf.linalg.cholesky_solve(Vchol, r[:, tf.newaxis])),
            axes=(-1, -1),
        )
        ldet = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(Vchol)))
        ldet2 = tf.math.log(
            tf.reduce_sum(
                tf.linalg.cholesky_solve(Vchol, tf.ones((self._ncells, 1), dtype=default_float()))
            )
        )

        return -ldet - 0.5 * quad - 0.5 * ldet2

    def _alphahat(self, Vchol):
        Vinvnu = tf.linalg.cholesky_solve(Vchol, self.nu[:, tf.newaxis])
        VinvX = tf.linalg.cholesky_solve(Vchol, tf.ones((self._ncells, 1), dtype=default_float()))
        return tf.reduce_sum(Vinvnu) / tf.reduce_sum(VinvX)

    def alphahat(self):
        return self._alphahat(tf.linalg.cholesky(self.V()))

    def _betahat(self, Vchol):
        return tf.squeeze(self.D() @ tf.linalg.cholesky_solve(Vchol, self._r(Vchol)[:, tf.newaxis]))

    def betahat(self):
        return self._betahat(tf.linalg.cholesky(self.V()))

    @property
    def nu(self):
        return (
            tf.math.log(self.muhat)
            + self.expression[:, self._currentgene] / self.muhat
            - 1
            - self._log_sizefactors
        )

    def _r(self, Vchol):
        return self.nu - self._alphahat(Vchol)

    def r(self):
        Vchol = tf.linalg.cholesky(self.V())
        return self._r(Vchol)

    @tf.function
    def estimate_muhat(self):
        Vchol = tf.linalg.cholesky(self.V())
        self.muhat.assign(
            tf.exp(self._alphahat(Vchol) + self._betahat(Vchol) + self._log_sizefactors)
        )

    def V(self):
        V = self.D()
        V = tf.linalg.set_diag(V, tf.linalg.diag_part(V) + 1 / self.muhat)
        return V

    # no property here, apparently tf.function has a problem with conditionals in properties
    def D(self):
        var = self.intrinsic_variance + self.environmental_variance
        var = tf.linalg.set_diag(var, tf.linalg.diag_part(var) + self.noise_variance)
        if self._use_interactions:
            var += self.interaction_variance
        return var

    def dV_dsigma(self):
        if self._use_interactions:
            return tf.stack(
                (
                    self.intrinsic_variance_matrix,
                    self.environmental_variance_matrix,
                    self.interaction_variance_matrix,
                    tf.eye(self._ncells, dtype=default_float()),
                ),
                axis=0,
            )
        else:
            return tf.stack(
                (
                    self.intrinsic_variance_matrix,
                    self.environmental_variance_matrix,
                    tf.eye(self._ncells, dtype=default_float()),
                ),
                axis=0,
            )

    def fraction_variance(self):
        intrinsic = gower_factor(self.intrinsic_variance)
        environ = gower_factor(self.environmental_variance)
        noise = self.noise_variance

        totalgower = intrinsic + environ + noise
        if self._use_interactions:
            interact = gower_factor(self.interaction_variance)
            totalgower += interact

            return self._fracvar_interact(
                (intrinsic / totalgower).numpy(),
                (environ / totalgower).numpy(),
                (interact / totalgower).numpy(),
                (noise / totalgower).numpy(),
            )
        else:
            return self._fracvar(
                (intrinsic / totalgower).numpy(),
                (environ / totalgower).numpy(),
                (noise / totalgower).numpy(),
            )

    @property
    def environmental_variance_matrix(self):
        return self.kernel.K(self._X)

    @property
    def interaction_variance_matrix(self):
        envmat = self.environmental_variance_matrix
        intmat = envmat @ tf.matmul(self.intrinsic_variance_matrix, envmat, transpose_b=True)
        return intmat / gower_factor(intmat)

    @property
    def intrinsic_variance(self):
        return self._sigmas[0] * self.intrinsic_variance_matrix

    @property
    def environmental_variance(self):
        return self._sigmas[1] * self.environmental_variance_matrix

    @property
    def interaction_variance(self):
        return self._sigmas[2] * self.interaction_variance_matrix

    @property
    def noise_variance(self):
        return self._sigmas[3]

    def use_interactions(self, interact: bool):
        self._old_interactions = self._use_interactions.numpy()
        self._use_interactions.assign(interact)
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._use_interactions.assign(self._old_interactions)

    def optimize(self, abstol: float = 1e-5, maxiter: int = 1000):
        oldsigmas = self._sigmas.numpy()
        for i in range(maxiter):
            self._opt.minimize()
            sigmas = self._sigmas.numpy()
            if np.all(np.abs(sigmas - oldsigmas) < abstol):
                break
            oldsigmas = sigmas
            self.estimate_muhat()


class SVCAInteractionScoreTest(ScoreTest):
    def __init__(
        self,
        expression_matrix: np.ndarray,
        X: np.ndarray,
        sizefactors: np.ndarray,
        kernel: Optional[gpflow.kernels.Kernel] = None,
    ):
        super().__init__()
        self._model = SVCA(expression_matrix, X, sizefactors, kernel)
        self._model.use_interactions(False)

    @property
    def kernel(self) -> List[gpflow.kernels.Kernel]:
        if self._model.kernel is not None:
            return [self._model.kernel]
        else:
            return []

    @kernel.setter
    def kernel(self, kernel: gpflow.kernels.Kernel):
        self._model.kernel = kernel

    def _fit_null(self, y):
        self._model.currentgene = y
        self._model.optimize()
        return None

    def _test(self, y, nullmodel: None):
        return self._do_test(
            self._model.r(),
            self._model.V(),
            self._model.dV_dsigma(),
            self._model.interaction_variance_matrix,
        )

    @staticmethod
    @tf.function(experimental_compile=True)
    def _do_test(residual, V, dV, interaction_mat):
        cholV = tf.linalg.cholesky(V)
        Vinvres = tf.squeeze(tf.linalg.cholesky_solve(cholV, residual[:, tf.newaxis]))
        stat = 0.5 * tf.tensordot(
            Vinvres, tf.tensordot(interaction_mat, Vinvres, axes=(-1, -1)), axes=(-1, -1)
        )

        Vinv_int = tf.linalg.cholesky_solve(cholV, interaction_mat)
        Vinv_dV = tf.linalg.cholesky_solve(cholV[tf.newaxis, ...], dV)

        Vinv_X = tf.squeeze(
            tf.linalg.cholesky_solve(
                cholV, tf.ones((tf.shape(residual)[0], 1), dtype=default_float())
            )
        )
        hatMat = Vinv_X[:, tf.newaxis] * Vinv_X[tf.newaxis, :] / tf.reduce_sum(Vinv_X)

        P_int = Vinv_int - hatMat @ interaction_mat
        P_dV = Vinv_dV - hatMat[tf.newaxis, ...] @ dV

        e_tilde = 0.5 * tf.linalg.trace(P_int)
        I_tau_tau = tf.reduce_sum(tf.transpose(P_int) * P_int)
        I_tau_theta = tf.reduce_sum(tf.transpose(P_int) * P_dV, axis=[-2, -1])
        I_theta_theta = tf.reduce_sum(
            tf.linalg.matrix_transpose(P_dV[tf.newaxis, ...]) * P_dV[:, tf.newaxis, ...],
            axis=[-2, -1],
        )

        I_tau_tau_tilde = 0.5 * (
            I_tau_tau
            - tf.tensordot(
                I_tau_theta,
                tf.squeeze(tf.linalg.solve(I_theta_theta, I_tau_theta[..., tf.newaxis])),
                axes=(-1, -1),
            )
        )

        return stat, e_tilde, I_tau_tau_tilde
