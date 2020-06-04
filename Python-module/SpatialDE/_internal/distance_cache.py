import tensorflow as tf
from gpflow.utilities.ops import square_distance, difference_matrix

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

