import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fgp = __import__('FaST-GP')


def get_coords(index):
    coords = pd.DataFrame(index=index)
    coords['x'] = index.str.split('x').str.get(0).map(float)
    coords['y'] = index.str.split('x').str.get(1).map(float)
    return coords


def main(l=5):
    df = pd.read_table('data/Rep7_MOB_count_matrix-1.tsv', index_col=0)
    sample_info = get_coords(df.index)
    
    X = sample_info[['x', 'y']]
    dfm = np.log10(df + 1)
    results_Rep7 = fgp.dyn_de(X, dfm, lengthscale=l)

    df = pd.read_table('data/Rep8_MOB_count_matrix-1.tsv', index_col=0)
    sample_info = get_coords(df.index)
    
    X = sample_info[['x', 'y']]
    dfm = np.log10(df + 1)
    results_Rep8 = fgp.dyn_de(X, dfm, lengthscale=l)

    idx = results_Rep7.index.intersection(results_Rep8.index)

    # plt.loglog()
    # plt.scatter(results_Rep7.loc[idx, 'max_delta'], results_Rep8.loc[idx, 'max_delta'])
    # plt.show()

    return results_Rep7.loc[idx], results_Rep8.loc[idx]


# if __name__ == '__main__':
#     main()
