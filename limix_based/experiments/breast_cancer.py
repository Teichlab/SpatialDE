import itertools
import logging

import pandas as pd
import NaiveDE
import numpy as np

from SpatialDE_limix.core.models import model_comparison

logger = logging.getLogger('limix_SpatialDE')
logger.setLevel('DEBUG')


def get_coords(index):
    coords = pd.DataFrame(index=index)
    coords['x'] = index.str.split('x').str.get(0).map(float)
    coords['y'] = index.str.split('x').str.get(1).map(float)
    return coords


def run():
    random_input = False
    gene_selection = range(1000)

    # preprocessing: same as in SpatialDE
    logger.info('Reading data')
    df = pd.read_table('../../Analysis/BreastCancer/data/Layer2_BC_count_matrix-1.tsv', index_col=0)
    df = df.T[df.sum(0) >= 3].T  # Filter practically unobserved genes
    sample_info = get_coords(df.index)
    sample_info['total_counts'] = df.sum(1)
    sample_info = sample_info.query('total_counts > 5')  # Remove empty features
    df = df.loc[sample_info.index]

    X = sample_info[['x', 'y']]
    dfm = NaiveDE.stabilize(df.T).T
    res = NaiveDE.regress_out(sample_info, dfm.T, 'np.log(total_counts)').T

    res = res.iloc[:, gene_selection]

    # specific code to limix_based implementation starts
    exp = res.values
    pos = X.values

    model1 = 'se_cor'
    model2 = 'se_no_cor'

    # single trait spatialDE analysis
    logger.info('Performing single-trait SpatialDE analysis')
    pval1, qval1 = model_comparison.run(pos, exp, model1, model2, P=1)

    gene_names = res.columns
    combinations =[i for i in itertools.combinations(gene_names,2)]

    header1 = ' '.join(gene_names)

    pd.DataFrame({'pval': pval1, 'qval': qval1}, index=gene_names).to_csv('bc_univariate.csv')


if __name__ == '__main__':
    run()
