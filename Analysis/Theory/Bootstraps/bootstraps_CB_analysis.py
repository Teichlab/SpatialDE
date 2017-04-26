import click
import numpy as np
import pandas as pd

import NaiveDE
import SpatialDE


def get_coords(index):
    coords = pd.DataFrame(index=index)
    coords['x'] = index.str.split('x').str.get(0).map(float)
    coords['y'] = index.str.split('x').str.get(1).map(float)
    return coords


@click.command()
@click.argument('out_file')
def main(out_file):
    df = pd.read_table('../../BreastCancer/data/Layer2_BC_count_matrix-1.tsv', index_col=0)
    df = df.T[df.sum(0) >= 3].T  # Filter practically unobserved genes
    sample_info = get_coords(df.index)
    sample_info['total_counts'] = df.sum(1)
    sample_info = sample_info.query('total_counts > 5')  # Remove empty features

    # Bootstrap sampling 80% of data
    sample_info = sample_info.sample(frac=0.8)

    df = df.loc[sample_info.index]

    X = sample_info[['x', 'y']]
    dfm = NaiveDE.stabilize(df.T).T
    res = NaiveDE.regress_out(sample_info, dfm.T, 'np.log(total_counts)').T

    results = SpatialDE.run(X, res)

    results.to_csv(out_file)

    return results 


if __name__ == '__main__':
    results = main()
