import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

from gpflow.base import Parameter
from gpflow.kernels import Kernel, Sum
from gpflow.utilities import positive
from gpflow.utilities.ops import square_distance
from gpflow.config import default_float


class Spectral(Kernel):
    def __init__(self, variance=1.0, lengthscale=1, period=1, **kwargs):
        super().__init__(**kwargs)
        self.variance = Parameter(variance, transform=positive())
        self.lengthscale = Parameter(lengthscale, transform=positive(lower=1e-6))
        self.period = Parameter(period, transform=positive(lower=1e-6))

        self._validate_ard_active_dims(self.lengthscale)
        self._validate_ard_active_dims(self.period)

    @property
    def ard(self) -> bool:
        return self.lengthscale.shape.ndims > 0

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X

        dist = X[:, tf.newaxis, :] - X2[tf.newaxis, :, :]
        cospart = tf.cos(2 * np.pi * tf.reduce_sum(dist / self.period, axis=-1))
        exppart = tf.reduce_prod(tf.exp(-2 * (np.pi * dist / self.lengthscale) ** 2), axis=-1)
        return self.variance * cospart * exppart

    def K_diag(self, X):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))

    def log_power_spectrum(self, s):
        s = tf.convert_to_tensor(s, dtype=self.variance.dtype)
        if s.ndim < 2:
            s = tf.expand_dims(s, 0)
        loc = tf.broadcast_to(self.period, (s.shape[1],))
        scale_diag = tf.broadcast_to(self.lengthscale, (s.shape[1],))
        mvd = tfp.distributions.MultivariateNormalDiag(loc=1/loc, scale_diag=1/scale_diag)
        return tf.math.log(tf.constant(0.5, dtype=self.variance.dtype)) + tf.reduce_logsumexp([mvd.log_prob(s), mvd.log_prob(-s)], axis=0)

class SpectralMixture(Sum):
    def __init__(self, kernels=None, **kwargs):
        if kernels is None:
            kernels = [Spectral()]
        elif isinstance(kernels, list):
            if not all(
            [isinstance(k, Spectral) for k in kernels]
        ):
                raise ValueError("Not all kernels are Spectral")
        else:
            kernels = [Spectral() for _ in range(kernels)]
        super().__init__(kernels)

    def log_power_spectrum(self, s):
        dens = []
        for k in self.kernels:
            dens.append(k.variance * k.log_power_spectrum(s))
        return tf.reduce_logsumexp(dens, axis=0)

    def plot_power_spectrum(self, xlim=None, ylim=None, **kwargs):
        if xlim is None or ylim is None:
            lengthscales = tf.convert_to_tensor([k.lengthscale for k in self.kernels])
            if lengthscales.ndim < 2:
                lengthscales = tf.tile(tf.expand_dims(lengthscales, axis=1), (1,2))
            periods = tf.convert_to_tensor([k.period for k in self.kernels])
            if periods.ndim < 2:
                periods = tf.tile(tf.expand_dims(periods, axis=1), (1,2))
            maxfreq = tf.math.argmin(periods, axis=0, output_type=tf.int32)
            maxfreq = tf.stack([maxfreq, tf.range(maxfreq.shape[0], dtype=tf.int32)], axis=1)
            limits = 1 / tf.gather_nd(periods, maxfreq)
            limits += 2 * tf.gather_nd(lengthscales, maxfreq)
        if xlim is None:
            xlim = limits[0]
        if ylim is None:
            ylim = limits[1]

        x, y = tf.meshgrid(tf.linspace(0, xlim, 1000), tf.linspace(0, ylim, 1000))
        ps = tf.reshape(self.log_power_spectrum(tf.stack([tf.reshape(x, (-1,)), tf.reshape(y, (-1,))], axis=1)), x.shape)

        fig, ax = plt.subplots()
        pos = ax.contourf(x, y, ps, levels=tf.linspace(tf.reduce_min(ps), tf.reduce_max(ps), 100), **kwargs)
        fig.colorbar(pos)
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
