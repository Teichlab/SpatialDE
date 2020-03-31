import tensorflow as tf
import tensorflow_probability as tfp

class TfpOptimizer:
    @staticmethod
    @tf.function
    def _concat_tensors(tens):
        return tf.concat([tf.reshape(t, (-1,)) for t in tens], axis=0)

    @staticmethod
    @tf.function
    def _assign_concat(x, vars):
        offset = 0
        for v in vars:
            newval = tf.reshape(x[offset : (offset + tf.size(v))], v.shape)
            v.assign(tf.where(tf.math.is_finite(newval), newval, v))
            offset += tf.size(v)

    @classmethod
    def _wrap_func(cls, func, vars):
        func = tf.function(func)

        def _objective(x):
            cls._assign_concat(x, vars)
            with tf.GradientTape() as t:
                obj = func()
            grads = cls._concat_tensors(t.gradient(obj, vars))
            return obj, grads

        return tf.function(_objective)

    @classmethod
    @tf.function
    def _bfgs(cls, func, vars, maxiter=1000, parallel_iterations=10, tol=1e-5):
        return tfp.optimizer.bfgs_minimize(
            func,
            cls._concat_tensors(vars),
            max_iterations=maxiter,
            parallel_iterations=parallel_iterations,
            tolerance=tol,
        )

    @classmethod
    @tf.function
    def _lbfgsb(cls, func, vars, maxiter=1000, parallel_iterations=10, tol=1e-5):
        return tfp.optimizer.lbfgs_minimize(
            func,
            cls._concat_tensors(vars),
            max_iterations=maxiter,
            parallel_iterations=parallel_iterations,
            tolerance=tol,
        )

    def minimize(
        self,
        closure,
        variables,
        method="bfgs",
        maxiter=1000,
        parallel_iterations=10,
        tol=1e-5,
    ):
        func = self._wrap_func(closure, variables)
        if method.lower() == "bfgs":
            res = self._bfgs(func, variables, maxiter, parallel_iterations, tol)
        elif method.lower == "l-bfgs-b":
            res = self._lbfgsb(func, variables, maxiter, parallel_iterations, tol)
        else:
            raise NotImplementedError
        self._assign_concat(res.position, variables)
        ret = {}
        ret["converged"] = res.converged
        ret["hess_inv"] = res.inverse_hessian_estimate
        ret["num_objective_evaluations"] = res.num_objective_evaluations
        return ret
