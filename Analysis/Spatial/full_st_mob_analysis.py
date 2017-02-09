import numpy as np
import pandas as pd

import fastgp as fgp


def get_coords(index):
    coords = pd.DataFrame(index=index)
    coords['x'] = index.str.split('x').str.get(0).map(float)
    coords['y'] = index.str.split('x').str.get(1).map(float)
    return coords


def main():
    df = pd.read_csv('data/Rep11_MOB_0.csv', index_col=0)
    sample_info = get_coords(df.index)
    
    X = sample_info[['x', 'y']]
    dfm = np.log10(df + 1)

    ks = {
        'PER': np.logspace(-1., np.log10(40), 10),
        'SE': np.logspace(-1., np.log10(40), 10),
        'linear': 0,
        'const': 0
    }
    results = fgp.dyn_de(X, dfm, kernel_space=ks)

    return results 


if __name__ == '__main__':
    results = main()
