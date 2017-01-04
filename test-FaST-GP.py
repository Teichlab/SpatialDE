import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

fgp = __import__('FaST-GP')


def get_coords(index):
    coords = pd.DataFrame(index=index)
    coords['x'] = index.str.split('x').str.get(0).map(float)
    coords['y'] = index.str.split('x').str.get(1).map(float)
    return coords

def main():
    df = pd.read_csv('data/Rep12_MOB_3.csv', index_col=0)
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
    df = pd.read_csv('data/Rep12_MOB_3.csv', index_col=0)
    sample_info = get_coords(df.index)

    X = sample_info[['x', 'y']]

    # example_genes = ['Nnat', 'Malsu1', 'Fmnl1']
    # dfm = np.log10(df + 1)[example_genes]

    dfm = np.log10(df + 1).sample(10, axis=1)
    l = 10

    K = fgp.SE_kernel(X, l)
    U, S = fgp.factor(K)
    UT1 = fgp.get_UT1(K)

    n, G = dfm.shape
    for g in range(G):
        y = dfm.iloc[:, g]
        UTy = fgp.get_UTy(U, y)
        LLs = []
        delta_range = np.logspace(base=np.e, start=-10, stop=10, num=100)
        max_ll = -np.inf
        max_delta = np.nan
        for delta in delta_range:
            cur_ll = fgp.LL(delta, UTy, UT1, S, n)
            LLs.append(cur_ll)
            if cur_ll > max_ll:
                max_ll = cur_ll
                max_delta = delta


        plt.subplot(np.ceil(G / 2.), 2, g + 1)
        plt.plot(delta_range, LLs, marker='o', markeredgecolor='w', markersize=2, markeredgewidth=1)
        plt.scatter([max_delta], [max_ll], marker='v', c='r', edgecolor='none', zorder=5)
        plt.title(dfm.columns[g])
        plt.xscale('log')
        plt.xlim(np.exp(-11), np.exp(11))

    plt.tight_layout()
    plt.savefig('example_grids.png', bbox_inches='tight')


if __name__ == '__main__':
    plot_LL_curves()
    # main()
