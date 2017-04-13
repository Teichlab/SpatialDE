import pandas as pd
import NaiveDE
import itertools
import numpy as np

from SpatialDE_limix.core.models import model_comparison

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
    pval1, qval1 = model_comparison.run(pos, exp, model1, model2, P=1)
    # two traits spatialDE analysis
    pval2, qval2 = model_comparison.run(pos, exp, model1, model2, P=2)

    gene_names = res.columns
    combinations =[i for i in itertools.combinations(gene_names,2)]

    header1 = ' '.join(gene_names)
    tmp = ['_'.join(elem) for elem in combinations]
    header2 = ' '.join(tmp)

    with open(res_dir + 'univariate.txt', 'w') as f:
        np.savetxt(f,
                   np.concatenate((pval1[None,:], qval1[None,:]), axis=0),
                   delimiter=' ',
                   header=header1,
                   fmt='%s',
                   comments='')

    with open(res_dir + 'bivariate.txt', 'w') as f:
        np.savetxt(f,
                   np.concatenate((pval2[None,:], qval2[None,:]), axis=0),
                   delimiter=' ',
                   header=header2,
                   fmt='%s',
                   comments='')


if __name__ == '__main__':
    run()
