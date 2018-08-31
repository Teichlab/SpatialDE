''' Wrapper functions to use SpatialDE directly on AnnData objects
'''
import pandas as pd

import NaiveDE

from .base import run
from .util import qvalue

def spatialde_test(adata, coord_columns=['x', 'y'], regress_formula='np.log(total_counts)'):
    ''' Run the SpatialDE test on an AnnData object

    Parameters
    ----------

    adata: An AnnData object with counts in the .X field.

    coord_columns: A list with the columns of adata.obs which represent spatial
                   coordinates. Default ['x', 'y'].

    regress_formula: A patsy formula for linearly regressing out fixed effects
                     from columns in adata.obs before fitting the SpatialDE models.
                     Default is 'np.log(total_counts)'.

    Returns
    -------

    results: A table of spatial statistics for each gene.
    '''
    adata.layers['stabilized'] = NaiveDE.stabilize(adata.X.T).T
    adata.layers['residual'] = NaiveDE.regress_out(adata.obs,
                                                   adata.layers['stabilized'].T,
                                                   regress_formula).T

    X = adata.obs[coord_columns].values
    expr_mat = pd.DataFrame.from_records(adata.layers['residual'],
                                         columns=adata.var.index,
                                         index=adata.obs.index)

    results = run(X, expr_mat)

    # Clip 0 pvalues
    min_pval = results.query('pval > 0')['pval'].min() / 2
    results['pval'] = results['pval'].clip_lower(min_pval)

    # Correct for multiple testing
    results['qval'] = qvalue(results['pval'], pi0=1.)

    return results
