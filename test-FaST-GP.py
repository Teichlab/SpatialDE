import logging

logging.basicConfig(level=logging.DEBUG)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

fgp = __import__('FaST-GP')
ds = __import__('data_simulation')


def get_coords(index):
    coords = pd.DataFrame(index=index)
    coords['x'] = index.str.split('x').str.get(0).map(float)
    coords['y'] = index.str.split('x').str.get(1).map(float)
    return coords

def main():
    df = pd.read_csv('data/Rep12_MOB_1.csv', index_col=0)
    sample_info = get_coords(df.index)

    # Run workflow
    X = sample_info[['x', 'y']]
    dfm = np.log10(df + 1)
    l = 10
    results = fgp.dyn_de(X, dfm, lengthscale=l, num=32)

    plt.scatter(results['max_delta'], results['max_ll'], c='k')
    plt.xscale('log')
    plt.xlim(np.exp(-11), np.exp(11))
    plt.xlabel('$\delta$')
    plt.ylabel('Maximum Log Likelihood')
    plt.title('lengthscale: {}'.format(l))
    plt.savefig('fastgp-fits.png', bbox_inches='tight')

    print(results.sort_values('max_delta').head(20))


def plot_LL_curves():
    # df = pd.read_csv('data/Rep12_MOB_3.csv', index_col=0)
    # sample_info = get_coords(df.index)
    # X = sample_info[['x', 'y']]
    # dfm = np.log10(df + 1).sample(10, axis=1)

    l = 10

    X, dfm, true_vals = ds.make_ls_data(l, 250, 10)
    true_vals['delta'] = true_vals['s2_e'] / true_vals['s2_t']

    K = fgp.SE_kernel(X, l)
    U, S = fgp.factor(K)
    UT1 = fgp.get_UT1(U)

    n, G = dfm.shape
    for g in range(G):
        y = dfm.iloc[:, g]
        UTy = fgp.get_UTy(U, y)
        LLs = []
        delta_range = np.logspace(base=np.e, start=-10, stop=10, num=32)
        max_ll = -np.inf
        max_delta = np.nan
        for delta in delta_range:
            cur_ll = fgp.LL(delta, UTy, UT1, S, n)
            LLs.append(cur_ll)
            if cur_ll > max_ll:
                max_ll = cur_ll
                max_delta = delta


        plt.subplot(np.ceil(G / 2.), 2, g + 1)
        plt.plot(delta_range, LLs, marker='o', markeredgecolor='w', markersize=2, markeredgewidth=1, c='k')
        plt.scatter([max_delta], [max_ll], marker='v', c='r', edgecolor='none', zorder=5)
        plt.title(dfm.columns[g])
        plt.axvline(true_vals.iloc[g, -1], color='r')
        plt.xscale('log')
        plt.xlim(np.exp(-11), np.exp(11))

    plt.savefig('example_grids.png')


def opt_simulation():
    l = 10
    logging.info('Sampling ground truth data...')
    X, dfm, true_vals = ds.make_ls_data(10, 500, 500)
    logging.info('Done')

    results = fgp.dyn_de(X, dfm, lengthscale=l, num=32)

    true_vals['delta'] = true_vals['s2_e'] / true_vals['s2_t']

    plt.subplot(3, 1, 1)
    plt.scatter(results['max_delta'], true_vals['delta'], c='k', label=None)
    plt.xscale('log')
    plt.xlim(np.exp(-11.), np.exp(11.))
    plt.yscale('log')
    plt.ylim(np.exp(-11.), np.exp(11.))
    plt.plot([1e-4, 1e4], [1e-4, 1e4], c='r', label='$ x = y $ line')

    plt.legend(loc='upper left')

    plt.ylabel('Ground truth $ \delta $')

    plt.subplot(3, 1, 2)
    plt.scatter(results['max_s2_t_hat'], true_vals['s2_t'], c='k')
    plt.xscale('log')
    plt.xlim(np.exp(-6.), np.exp(6.))
    plt.yscale('log')
    plt.ylim(np.exp(-6.), np.exp(6.))
    plt.plot([1e-2, 1e2], [1e-2, 1e2], c='r')
    plt.ylabel('Ground truth $ \sigma_t^2 $')

    plt.subplot(3, 1, 3)
    plt.scatter(results['max_mu_hat'], true_vals['mu'], c='k')
    plt.xlim(-1, 6)
    plt.ylim(-1, 6)
    plt.plot([0, 5], [0, 5], c='r')
    plt.ylabel('Ground truth $ \mu $')

    plt.xlabel('Inferred Value')

    plt.savefig('simulation_accuracy.png')


if __name__ == '__main__':
    opt_simulation()
    # plot_LL_curves()
    # main()
