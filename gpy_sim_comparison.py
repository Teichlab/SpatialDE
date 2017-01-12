from time import time

import numpy as np
from tqdm import tqdm
import GPy
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import cm

import GPflow

fgp = __import__('FaST-GP')
ds = __import__('data_simulation')


def dyn_de_gpf(X, exp_tab, lengthscale=10):
    results = []
    n, G = exp_tab.shape
    for g in tqdm(range(G)):
        y = exp_tab.iloc[:, [g]].as_matrix()
        kernel = GPflow.kernels.RBF(1, lengthscales=lengthscale)
        kernel.lengthscales.fixed = True
        mf = GPflow.mean_functions.Constant()
        gpr = GPflow.gpr.GPR(X, y, kern=kernel, mean_function=mf)
        t0 = time()
        o = gpr.optimize()
        t = time() - t0

        results.append({'g': exp_tab.columns[g],
                        'max_ll': 0,
                        'max_delta': gpr.likelihood.variance.value[0] / gpr.kern.variance.value[0],
                        'max_mu_hat': gpr.mean_function.c.value[0],
                        'max_s2_t_hat': gpr.kern.variance.value[0],
                        'time': t})

    return pd.DataFrame(results)    


def dyn_de_gpy(X, exp_tab, lengthscale=10):
    y = exp_tab.iloc[:, [0]]
    kernel = GPy.kern.RBF(1, lengthscale=lengthscale)
    kernel.lengthscale.fix()
    mf = GPy.mappings.Constant(1, 1)
    gpr = GPy.models.GPRegression(X, y, kernel=kernel, mean_function=mf)

    results = []
    n, G = exp_tab.shape
    for g in tqdm(range(G)):
        y = exp_tab.iloc[:, [g]]
        gpr.set_Y(y)
        gpr[:] = 1.0
        gpr.kern.lengthscale.fix(lengthscale)
        t0 = time()
        gpr.optimize()
        t = time() - t0

        results.append({'g': exp_tab.columns[g],
                        'max_ll': -gpr.objective_function(),
                        'max_delta': gpr.likelihood.variance[0] / gpr.rbf.variance[0],
                        'max_mu_hat': gpr.constmap.C[0],
                        'max_s2_t_hat': gpr.rbf.variance[0],
                        'time': t})

    return pd.DataFrame(results)


def opt_simulation_inference_accuracy():
    l = 10
    X, dfm, true_vals = ds.make_ls_data(l, 500, 25)
    true_vals['delta'] = true_vals['s2_e'] / true_vals['s2_t']

    
    gpf_results = dyn_de_gpf(X, dfm, lengthscale=l)

    gpy_results = dyn_de_gpy(X, dfm, lengthscale=l)

    fgp_results = fgp.dyn_de(X, dfm, lengthscale=l)

    plt.xscale('log')
    plt.yscale('log')

    p_error = np.abs(true_vals['s2_t'].as_matrix() - gpf_results['max_s2_t_hat'].as_matrix())
    plt.scatter(gpf_results['time'], p_error, c='b', label='GPFlow', edgecolor='none')

    p_error = np.abs(true_vals['s2_t'].as_matrix() - gpy_results['max_s2_t_hat'].as_matrix())
    plt.scatter(gpy_results['time'], p_error, c='k', label='GPy', edgecolor='none')

    p_error = np.abs(true_vals['s2_t'].as_matrix() - fgp_results['max_s2_t_hat'].as_matrix())
    plt.scatter(fgp_results['time'], p_error, c='r', label='FGP', edgecolor='none')

    plt.legend()
    plt.savefig('sim_inference_accuracy.png')


def make_diff_cell_simulation_data(data_sizes=[50, 100, 250, 500, 1000]):
    l = 10.
    for N in data_sizes:
        X, dfm, true_vals = ds.make_ls_data(l, N, 25)
        pd.DataFrame(X).to_csv('sim_data/X_{}.csv'.format(N))
        dfm.to_csv('sim_data/dfm_{}.csv'.format(N))
        true_vals.to_csv('sim_data/true_vals_{}.csv'.format(N))


def make_diff_ls_simulation_data():
    X, dfm, true_vals = ds.make_multi_ls_data()
    pd.DataFrame(X).to_csv('sim_data/X_multi_ls.csv')
    dfm.to_csv('sim_data/dfm_multi_ls.csv')
    true_vals.to_csv('sim_data/true_vals_multi_ls.csv')


def compare_inference_speeds():
    from glob import glob

    Ns = []
    results_fgp = []
    results_gpy = []
    
    for dfm_file in glob('sim_data/dfm_*.csv'):
        dfm = pd.read_csv(dfm_file, index_col=0)
        N = dfm.shape[0]
        Ns.append(N)
        X = pd.read_csv(dfm_file.replace('dfm', 'X'), index_col=0).as_matrix()

        t0 = time()
        dyn_de_gpy(X, dfm, lengthscale=10)
        t = time() - t0
        results_gpy.append(t)

        t0 = time()
        fgp.dyn_de(X, dfm, lengthscale=10)
        t = time() - t0
        results_fgp.append(t)

    # plt.xscale('log')
    plt.yscale('log')
    plt.scatter(Ns, results_gpy, c='k', s=50, label='GPy')
    plt.scatter(Ns, results_fgp, c='r', s=50, label='FaST-GP')
    
    plt.legend(loc='upper left')
    plt.title('Inference time for 25 genes')
    plt.xlabel('# Cells')
    plt.ylabel('Time (seconds)')

    plt.savefig('sim_speed.png')
    

def identify_lengthscale():
    dfm = pd.read_csv('sim_data/dfm_multi_ls.csv', index_col=0)
    X = pd.read_csv('sim_data/X_multi_ls.csv', index_col=0).as_matrix()

    ks = {
        'SE': np.logspace(0., 2., 10),
        'linear': 0,
        'const': 0,
        'null': 0
    }
    results = fgp.dyn_de(X, dfm, kernel_space=ks)
    results = results[results.groupby(['g'])['BIC'].transform(min) == results['BIC']]

    true_vals = pd.read_csv('sim_data/true_vals_multi_ls.csv', index_col=0)
    true_vals['delta'] = true_vals['s2_e'] / true_vals['s2_t']

    return results, true_vals


if __name__ == '__main__':
    # opt_simulation_inference_accuracy()
    # make_diff_cell_simulation_data()
    # compare_inference_speeds()
    # make_diff_ls_simulation_data()
    results, true_vals = identify_lengthscale()

    plt.figure()
    plt.loglog()
    plt.scatter(results['l'], true_vals.loc[results['g'], 'l'], c=np.log10(true_vals['delta']),
                cmap=cm.magma, edgecolor='none', s=30)
    plt.colorbar()
    plt.savefig('inferred_lengthscales.png')

    plt.figure()
    plt.loglog()
    plt.scatter(results['max_delta'], true_vals.loc[results['g'], 'delta'], c=np.log10(true_vals['l']),
                cmap=cm.magma, edgecolor='none', s=30)
    plt.colorbar()
    plt.savefig('inferred_delta.png')

    print(results.model.value_counts())
