from typing import Union, Optional
from abc import ABCMeta, abstractmethod

import numpy as np
import scipy

def distance(X: np.ndarray, Y: Optional[np.ndarray]=None):
    if Y is None:
        Y = X
    return X[:, np.newaxis, :] - Y[np.newaxis, ...]

def squared_distance(X: np.ndarray, Y: Optional[np.ndarray]=None):
    Xs = np.sum(np.square(X), axis=-1, keepdims=True)
    if Y is None:
        Y = X
        Ys = Xs
    else:
        Ys = np.sum(np.square(Y), axis=-1, keepdims=True)

    dist = -2 * X @ Y.T
    return dist + Xs + Ys.T

def scale(X:np.ndarray, lengthscale: Optional[float] = 1):
    if X is not None:
        return X / lengthscale
    else:
        return X

def scaled_distance(
    X: np.ndarray, Y: Optional[np.ndarray] = None, lengthscale: Optional[float] = 1,
):
    return distance(scale(X, lengthscale), scale(Y, lengthscale))

def scaled_squared_distance(X: np.ndarray, Y: Optional[np.ndarray] = None, lengthscale: Optional[float] = 1):
    return squared_distance(scale(X, lengthscale), scale(Y, lengthscale))

class Kernel():
    def K(self, X: np.ndarray, Y: Optional[np.ndarray] = None):
        pass

    def K_diag(self, X: np.ndarray):
        pass

class StationaryKernel(Kernel):
    def __init__(self, lengthscale=1):
        self.lengthscale=lengthscale

class SquaredExponential(StationaryKernel):
    def K(self,
        X: np.ndarray, Y: Optional[np.ndarray] = None
    ):
        return np.exp(-0.5 * scaled_squared_distance(X, Y, self.lengthscale))

    def K_diag(self, X: np.ndarray):
        return np.repeat(1, X.shape[0])


class Cosine(StationaryKernel):
    def K(self,
        X: np.ndarray, Y: Optional[np.ndarray] = None,
    ):
        return np.cos(2 * np.pi * np.sum(scaled_distance(X, Y, self.lengthscale), axis=-1))

    def K_diag(self, X: np.ndarray):
        return np.repeat(1, X.shape[0])


class Linear(Kernel):
    def K(self, X: np.ndarray, Y: Optional[np.ndarray] = None):
        if Y is None:
            Y = X
        return np.sum(X[:, np.newaxis, :] * Y[np.newaxis, ...], axis=-1)

    def K_diag(self, X: np.ndarray):
        return np.sum(X ** 2, axis=-1)
