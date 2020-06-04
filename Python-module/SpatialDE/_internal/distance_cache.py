import tensorflow as tf
from gpflow.utilities.ops import square_distance, difference_matrix

class DistanceCache:
    dtype = tf.float64
    def __init__(self, X: tf.Tensor):
        self.X = tf.convert_to_tensor(X, dtype=self.dtype)
        self._squaredEuclidean = None
        self._sumDiff = None

    @property
    def squaredEuclideanDistance(self):
        if self._squaredEuclidean is None:
            self._squaredEuclidean = square_distance(tf.cast(self.X, dtype=self.dtype), None)
        return self._squaredEuclidean

    @property
    def sumOfDifferences(self):
        if self._sumDiff is None:
            self._sumDiff = tf.reduce_sum(difference_matrix(tf.cast(self.X, dtype=self.dtype), None), axis=-1)
        return self._sumDiff

