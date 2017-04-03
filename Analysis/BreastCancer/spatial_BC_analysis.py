import numpy as np
import pandas as pd

import NaiveDE
import SpatialDE


def get_coords(index):
    coords = pd.DataFrame(index=index)
    coords['x'] = index.str.split('x').str.get(0).map(float)
    coords['y'] = index.str.split('x').str.get(1).map(float)
    return coords


def main():
    df = pd.read_table('data/Layer2_BC_count_matrix-1.tsv', index_col=0)
    df = df.T[df.sum(0) >= 3].T  # Filter practically unobserved genes
    sample_info = get_coords(df.index)
    sample_info['total_counts'] = df.sum(1)
    sample_info = sample_info.query('total_counts > 5')  # Remove empty features
    df = df.loc[sample_info.index]

    X = sample_info[['x', 'y']]
    dfm = NaiveDE.stabilize(df.T).T
    res = NaiveDE.regress_out(sample_info, dfm.T, 'np.log(total_counts)').T

    results = SpatialDE.run(X, res)

    sample_info.to_csv('BC_sample_info.csv')
    results.to_csv('BC_final_results.csv')

    de_results = results[(results.qval < 0.05)].copy()
    ms_results = SpatialDE.model_search(X, res, de_results)

    ms_results.to_csv('BC_MS_results.csv')

    return results 


if __name__ == '__main__':
    results = main()
