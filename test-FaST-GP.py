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

@profile
def main():
    df = pd.read_csv('data/Rep11_MOB_3.csv', index_col=0)
    sample_info = get_coords(df.index)

    # Run workflow
    l = 25
    K = fgp.SE_kernel(sample_info[['x', 'y']], l)
    U, S = fgp.factor(K)
    UT1 = fgp.get_UT1(U)

    results = []
    dfm = np.log10(df.as_matrix() + 1)
    n = dfm.shape[0]
    for g in tqdm(range(n)):
        y = dfm[:, g]

        UTy = fgp.get_UTy(U, y)

        max_ll = -np.inf
        max_delta = np.nan
        for delta in np.logspace(base=np.e, start=-10, stop=10, num=100):
            cur_ll = fgp.LL(delta, UTy, UT1, S, n)
            if cur_ll > max_ll:
                max_ll = cur_ll
                max_delta = delta

        # print(g, max_ll, max_delta)
    #     results.append({'g': g, 'max_ll': max_ll, 'max_delta': max_delta})

    # results = pd.DataFrame(results)
    # plt.scatter(results['max_delta'], results['max_ll'], c='k')
    # plt.xscale('log')
    # plt.xlim(np.exp(-10), np.exp(10))
    # plt.show()

if __name__ == '__main__':
    main()
