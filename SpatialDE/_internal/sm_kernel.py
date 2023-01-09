import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

from gpflow import Parameter
from gpflow.kernels import Stationary, Sum
from gpflow.utilities import positive
from gpflow.utilities.ops import square_distance, difference_matrix
from gpflow.utilities import to_default_float


class Spectral(Stationary):
    def __init__(self, variance=1.0, lengthscales=1, periods=1, **kwargs):
        super().__init__(
            variance=variance,
            lengthscales=Parameter(lengthscales, transform=positive(lower=to_default_float(1e-6))),
            **kwargs,
        )
        self.periods = Parameter(periods, transform=positive(lower=to_default_float(1e-6)))

        self._validate_ard_active_dims(self.periods)

    @property
    def ard(self) -> bool:
        return self.lengthscales.shapes.ndims > 0 or self.periods.shapes.ndims > 0

    def K(self, X, X2=None):
        return self.variance * self.K_novar(X, X2)

    def K_novar(self, X, X2=None):
        Xs = X / self.periods
        if X2 is not None:
            X2s = X2 / self.periods
        else:
            X2s = None
        dist = difference_matrix(Xs, X2s)
        cospart = tf.cos(2 * np.pi * tf.reduce_sum(dist, axis=-1))

        dist = square_distance(self.scale(X), self.scale(X2))
        exppart = tf.exp(-0.5 * dist)

        return cospart * exppart

    def K_diag(self, X):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))

    def log_power_spectrum(self, s):
        if not tf.is_tensor(s):
            s = tf.convert_to_tensor(s, dtype=self.variance.dtype)
        else:
            s = tf.cast(s, dtype=self.variance.dtype)
        if s.ndim < 2:
            s = tf.expand_dims(s, 1)
        loc = tf.broadcast_to(self.periods, (s.shape[1],))
        scale_diag = tf.broadcast_to(self.lengthscales / (0.25 * np.pi**2), (s.shape[1],))
        mvd = tfp.distributions.MultivariateNormalDiag(loc=1 / loc, scale_diag=1 / scale_diag)
        return tf.math.log(tf.constant(0.5, dtype=self.variance.dtype)) + tf.reduce_logsumexp(
            [mvd.log_prob(s), mvd.log_prob(-s)], axis=0
        )


class SpectralMixture(Sum):
    def __init__(self, kernels=None, dimnames=None, **kwargs):
        if kernels is None:
            kernels = [Spectral()]
        elif isinstance(kernels, list):
            if not all([isinstance(k, Spectral) for k in kernels]):
                raise ValueError("Not all kernels are Spectral")
        else:
            kernels = [Spectral() for _ in range(kernels)]
        super().__init__(kernels)

        if dimnames is None:
            self.dimnames = ("X", "Y")
        else:
            self.dimnames = dimnames

    def K_novar(self, X, X2=None):
        return self._reduce([k.K_novar(X, X2) for k in self.kernels])

    def log_power_spectrum(self, s):
        dens = []
        for k in self.kernels:
            dens.append(k.variance * k.log_power_spectrum(s))
        return tf.reduce_logsumexp(dens, axis=0)

    def plot_power_spectrum(self, xlim: float = None, ylim: float = None, **kwargs):
        if xlim is None or ylim is None:
            lengthscales = tf.convert_to_tensor([k.lengthscales for k in self.kernels])
            if lengthscales.ndim < 2:
                lengthscales = tf.tile(tf.expand_dims(lengthscales, axis=1), (1, 2))
            periods = tf.convert_to_tensor([k.periods for k in self.kernels])
            if periods.ndim < 2:
                periods = tf.tile(tf.expand_dims(periods, axis=1), (1, 2))
            maxfreq = tf.math.argmin(periods, axis=0, output_type=tf.int32)
            maxfreq = tf.stack([maxfreq, tf.range(maxfreq.shape[0], dtype=tf.int32)], axis=1)
            limits = 1 / tf.gather_nd(periods, maxfreq)
            limits += 2 * tf.gather_nd(lengthscales, maxfreq)
        if xlim is None:
            xlim = limits[0].numpy()
        else:
            xlim = np.asarray([xlim])[0]
        if ylim is None:
            ylim = limits[1].numpy()
        else:
            ylim = np.asarray([ylim])[0]

        limtype = np.promote_types(xlim.dtype, ylim.dtype)
        xlim = xlim.astype(limtype)
        ylim = ylim.astype(limtype)

        dim = max(
            [k.lengthscales.ndim for k in self.kernels] + [k.periods.ndim for k in self.kernels]
        )
        fig, ax = plt.subplots()
        if dim < 1:
            x = tf.linspace(0, xlim, 1000)
            ps = self.log_power_spectrum(x)
            ax.plot(x, ps)
            ax.set_xlim((0, xlim.numpy()))
            ax.set_xlabel("frequency")
            ax.set_ylabel("log spectral density")
        else:
            x, y = tf.meshgrid(tf.linspace(0.0, xlim, 1000), tf.linspace(0.0, ylim, 1000))
            ps = tf.reshape(
                self.log_power_spectrum(
                    tf.stack([tf.reshape(x, (-1,)), tf.reshape(y, (-1,))], axis=1)
                ),
                x.shape,
            )
            pos = ax.contourf(
                x, y, ps, levels=tf.linspace(tf.reduce_min(ps), tf.reduce_max(ps), 100), **kwargs
            )
            ax.set_xlabel(f"{self.dimnames[0]} frequency")
            ax.set_ylabel(f"{self.dimnames[1]} frequency")
            cbar = fig.colorbar(pos)
            cbar.ax.set_ylabel("log spectral density")
        return ax

    def __iter__(self):
        self._i = 0
        return self

    def __next__(self):
        if self._i < len(self.kernels):
            k = self.kernels[self._i]
            self._i += 1
            return k
        else:
            raise StopIteration
