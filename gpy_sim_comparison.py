import numpy as np
from tqdm import tqdm
import GPy

ds = __import__('data_simulation')

def opt_simulation():
    l = 10
    X, dfm, true_vals = ds.make_ls_data(10, 500, 10)
    true_vals['delta'] = true_vals['s2_e'] / true_vals['s2_t']

    y = dfm.iloc[:, [0]]
    kernel = GPy.kern.RBF(1, lengthscale=l)
    kernel.lengthscale.fix()
    mf = GPy.mappings.Constant(1, 1)
    gpr = GPy.models.GPRegression(X, y, kernel=kernel, mean_function=mf)

    results = []
    n, G = dfm.shape
    for g in range(G):
        y = dfm.iloc[:, [g]]
        gpr.set_Y(y)
        gpr[:] = 1.0
        gpr.kern.lengthscale.fix(l)
        gpr.optimize()

        print(gpr.likelihood.variance / gpr.rbf.variance, true_vals.delta.iloc[g])

if __name__ == '__main__':
    opt_simulation()