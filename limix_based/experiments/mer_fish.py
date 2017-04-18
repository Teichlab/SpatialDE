import itertools
import logging

import pandas as pd
import NaiveDE
import numpy as np

from SpatialDE_limix.core.models import model_comparison

logger = logging.getLogger('limix_SpatialDE')
logger.setLevel('DEBUG')

def run():
    # preprocessing: same as in SpatialDE
    data_dir = 'merfish_data/'
    res_dir = 'merfish_res/'
    random_input = True
    gene_selection = range(3)

    df = pd.read_csv(data_dir + '/middle_exp_mat.csv', index_col=0)
    df = df.T[df.sum(0) >= 3].T  # Filter practically unobserved genes

    # Get coordinates for each sample
    sample_info = pd.read_csv(data_dir + '/middle_sample_info.csv', index_col=0)
    df = df.loc[sample_info.index]

    X = sample_info[['abs_X', 'abs_Y']]

    # Convert data to log-scale, and account for depth
    dfm = NaiveDE.stabilize(df.T).T
    res = NaiveDE.regress_out(sample_info, dfm.T, 'np.log(cytoplasmArea)').T

    res = res.iloc[:, gene_selection]

    # specific code to limix_based implementation starts
    exp = res.values
    pos = X.values

    if random_input:
        pos = pos[np.random.choice(range(pos.shape[0]), pos.shape[0], replace=False), :]
        np.random.shuffle(exp)

    model1 = 'se_cor'
    model2 = 'se_no_cor'

    # single trait spatialDE analysis
    logger.info('Performing single-trait SpatialDE analysis')
    pval1, qval1 = model_comparison.run(pos, exp, model1, model2, P=1)

    # two traits spatialDE analysis
    logger.info('Performing two-trait SpatialDE analysis')
    pval2, qval2 = model_comparison.run(pos, exp, model1, model2, P=2)

    gene_names = res.columns
    combinations =[i for i in itertools.combinations(gene_names,2)]

    header1 = ' '.join(gene_names)
    tmp = ['_'.join(elem) for elem in combinations]
    header2 = ' '.join(tmp)

    pd.DataFrame({'pval': pval1, 'qval': qval1}, index=gene_names).to_csv(res_dir + 'univariate.csv')
    pd.DataFrame({'pval': pval2, 'qval': qval2}, index=tmp).to_csv(res_dir + 'bivariate.csv')


if __name__ == '__main__':
    run()
