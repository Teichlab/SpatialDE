from typing import Union, Optional
from abc import ABCMeta, abstractmethod
import math

import tensorflow as tf
from gpflow.utilities.ops import square_distance, difference_matrix


def scale(X: tf.Tensor, lengthscale: Optional[float] = 1):
    if X is not None:
        return X / lengthscale
    else:
        return X


def scaled_difference_matrix(
    X: tf.Tensor, Y: Optional[tf.Tensor] = None, lengthscale: Optional[float] = 1,
):
    return difference_matrix(scale(X, lengthscale), scale(Y, lengthscale))


def scaled_squared_distance(
    X: tf.Tensor, Y: Optional[tf.Tensor] = None, lengthscale: Optional[float] = 1
):
    return square_distance(scale(X, lengthscale), scale(Y, lengthscale))


class DistanceCache:
    def __init__(self, X):
        self.X = X
        self._squaredEuclidean = None
        self._sumDiff = None

    @property
    def squaredEuclideanDistance(self):
        if self._squaredEuclidean is None:
            self._squaredEuclidean = square_distance(self.X, None)
        return self._squaredEuclidean

    @property
    def sumOfDifferences(self):
        if self._sumDiff is None:
            self._sumDiff = tf.reduce_sum(difference_matrix(self.X, None), axis=-1)
        return self._sumDiff

class Kernel:
    def __init__(self, cache: DistanceCache):
        self._cache = cache

    def K(self, X: tf.Tensor, Y: Optional[tf.Tensor] = None):
        pass

    def K_diag(self, X: tf.Tensor):
        pass


class StationaryKernel(Kernel):
    def __init__(self, cache: DistanceCache, lengthscale=1):
        super().__init__(cache)
        self.lengthscale = lengthscale


class SquaredExponential(StationaryKernel):
    def K(self, X: Optional[tf.Tensor] = None, Y: Optional[tf.Tensor] = None):
        if X is None and Y is not None:
            X, Y = Y, X
        if (X is None or X is self._cache.X) and Y is None:
            dist = self._cache.squaredEuclideanDistance / self.lengthscale ** 2
        else:
            dist = scaled_squared_distance(X, Y, self.lengthscale)
        return tf.exp(-0.5 * dist)

    def K_diag(self, X: tf.Tensor):
        return tf.repeat(1, X.shape[0])


class Cosine(StationaryKernel):
    def K(
        self, X: Optional[tf.Tensor] = None, Y: Optional[tf.Tensor] = None,
    ):
        if X is None and Y is not None:
            X, Y = Y, X
        if (X is None or X is self._cache.X) and Y is None:
            dist = self._cache.sumOfDifferences / self.lengthscale
        else:
            dist = tf.reduce_sum(scaled_difference_matrix(X, Y, self.lengthscale), axis=-1)
        return tf.cos(2 * math.pi * dist)

    def K_diag(self, X: tf.Tensor):
        return tf.repeat(1, X.shape[0])


class Linear(Kernel):
    def K(self, X: tf.Tensor, Y: Optional[tf.Tensor] = None):
        if Y is None:
            Y = X
        return tf.sum(X[:, tf.newaxis, :] * Y[tf.newaxis, ...], axis=-1)

    def K_diag(self, X: tf.Tensor):
        return tf.sum(tf.square(X), axis=-1)
