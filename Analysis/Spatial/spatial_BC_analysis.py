import numpy as np
import pandas as pd

import SpatialDE as sde


def get_coords(index):
    coords = pd.DataFrame(index=index)
    coords['x'] = index.str.split('x').str.get(0).map(float)
    coords['y'] = index.str.split('x').str.get(1).map(float)
    return coords


def main():
    df = pd.read_table('data/Layer2_BC_count_matrix-1.tsv', index_col=0)
    df = df.T[df.sum(0) >= 3].T  # Filter practically unobserved genes
    sample_info = get_coords(df.index)

    X = sample_info[['x', 'y']]
    dfm = np.log10(df + 1)

    results = sde.run(X, dfm)

    sample_info.to_csv('BC_sample_info.csv')
    results.to_csv('BC_final_results.csv')

    return results 


if __name__ == '__main__':
    results = main()
