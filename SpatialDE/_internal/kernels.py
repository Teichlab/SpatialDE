from typing import Union, Optional
from abc import ABCMeta, abstractmethod
import math

import tensorflow as tf
from gpflow import default_float
from gpflow.utilities import to_default_float
from gpflow.utilities.ops import square_distance, difference_matrix

from .distance_cache import DistanceCache


def scale(X: tf.Tensor, lengthscale: Optional[float] = 1):
    if X is not None:
        return X / lengthscale
    else:
        return X


def scaled_difference_matrix(
    X: tf.Tensor,
    Y: Optional[tf.Tensor] = None,
    lengthscale: Optional[float] = 1,
):
    return difference_matrix(scale(X, lengthscale), scale(Y, lengthscale))


def scaled_squared_distance(
    X: tf.Tensor, Y: Optional[tf.Tensor] = None, lengthscale: Optional[float] = 1
):
    return square_distance(scale(X, lengthscale), scale(Y, lengthscale))


class Kernel(metaclass=ABCMeta):
    def __init__(self, cache: DistanceCache):
        self._cache = cache

    def K(self, X: Optional[tf.Tensor] = None, Y: Optional[tf.Tensor] = None):
        if X is None and Y is not None:
            X, Y = Y, X
        if (X is None or X is self._cache.X) and Y is None:
            return self._K(cache=True)
        else:
            X = to_default_float(X)
            if Y is not None:
                Y = to_default_float(Y)
        return self._K(X, Y)

    def K_diag(self, X: Optional[tf.Tensor]):
        if X is None:
            return self._K_diag(cache=True)
        else:
            return self._K_diag(to_default_float(X))

    @abstractmethod
    def _K(self, X: Optional[tf.Tensor] = None, Y: Optional[tf.Tensor] = None, cache: bool = False):
        pass

    @abstractmethod
    def _K_diag(self, X: Optional[tf.Tensor] = None, cache: bool = False):
        pass


class StationaryKernel(Kernel):
    def __init__(self, cache: DistanceCache, lengthscale=1):
        super().__init__(cache)
        self.lengthscale = lengthscale

    def _K_diag(self, X: Optional[tf.Tensor] = None, cache: bool = False):
        if cache:
            n = self._cache.X.shape[0]
        else:
            n = X.shape[0]
        return self._K_diag_impl(n)

    def _K_diag_impl(self, n: int):
        return tf.repeat(tf.convert_to_tensor(1, dtype=default_float()), n)


class SquaredExponential(StationaryKernel):
    def _K(self, X: Optional[tf.Tensor] = None, Y: Optional[tf.Tensor] = None, cache: bool = False):
        if cache:
            dist = self._cache.squaredEuclideanDistance / self.lengthscale**2
        else:
            dist = scaled_squared_distance(X, Y, self.lengthscale)
        return tf.exp(-0.5 * dist)


class Cosine(StationaryKernel):
    def _K(self, X: Optional[tf.Tensor] = None, Y: Optional[tf.Tensor] = None, cache: bool = False):
        if cache:
            dist = self._cache.sumOfDifferences / self.lengthscale
        else:
            dist = tf.reduce_sum(scaled_difference_matrix(X, Y, self.lengthscale), axis=-1)
        return tf.cos(2 * math.pi * dist)


class Linear(Kernel):
    def _K(self, X: tf.Tensor, Y: Optional[tf.Tensor] = None):
        if Y is None:
            Y = X
        return tf.sum(X[:, tf.newaxis, :] * Y[tf.newaxis, ...], axis=-1)

    def _K_diag(self, X: tf.Tensor, cache: bool = False):
        if cache:
            X = self._cache.X
        return tf.sum(tf.square(X), axis=-1)
