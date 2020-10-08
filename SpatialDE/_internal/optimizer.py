import scipy.optimize
import tensorflow as tf

from .util import concat_tensors, assign_concat


class MultiScipyOptimizer:
    def __init__(self, objective, variables):
        self.objective = objective
        self.variables = variables
        self._obj = self._wrap_func(objective, variables)

    def minimize(self, method="bfgs", **scipy_kwargs):
        res = scipy.optimize.minimize(
            self._obj,
            concat_tensors(self.variables).numpy(),
            method=method,
            jac=True,
            **scipy_kwargs,
        )
        assign_concat(res.x, self.variables)
        return res

    @classmethod
    def _wrap_func(cls, func, vars):
        def _objective(x):
            assign_concat(x, vars)
            with tf.GradientTape() as t:
                obj = func()
            grads = concat_tensors(
                t.gradient(obj, vars, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            )
            return obj, grads

        _objective = tf.function(_objective)

        def _obj(x):
            loss, grad = _objective(x)
            return loss.numpy(), grad.numpy()

        return _obj
