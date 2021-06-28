import functools
import tensorflow as tf
from gpflow import default_float
from gpflow.utilities import to_default_float
from gpflow.utilities.ops import square_distance, difference_matrix


def cached(variable):
    def cache(func):
        func = tf.function(func, experimental_compile=True, experimental_relax_shapes=True)

        @functools.wraps(func)
        def caching_wrapper(self, *args, **kwargs):
            if not hasattr(self, variable) or getattr(self, variable) is None:
                mat = func(self.X)
                if self._cache:
                    setattr(self, variable, mat)
            else:
                mat = getattr(self, variable)
            return mat

        return caching_wrapper

    return cache


class DistanceCache:
    def __init__(self, X: tf.Tensor, cache=True):
        self.X = X
        self._cache = cache

    @property
    @cached("_squaredEuclidean")
    def squaredEuclideanDistance(X):
        return square_distance(to_default_float(X), None)

    @property
    @cached("_sumDiff")
    def sumOfDifferences(X):
        return tf.reduce_sum(difference_matrix(to_default_float(X), None), axis=-1)
