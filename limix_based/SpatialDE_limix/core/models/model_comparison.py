import numpy as np

# from here
from SpatialDE_limix.core.models import null_gp, se_spatial_no_cor_gp, se_spatial_gp
from SpatialDE_limix.core.utils import util


def run_model(X, Y, model, P):
    if model == 'se_cor':
        gps = se_spatial_gp(X, Y, P)
        gps.optimize_all()
        return gps

    elif model == 'se_no_cor':
        gps = se_spatial_no_cor_gp(X, Y, P)
        gps.optimize_all()
        return gps

    elif model == 'null':
        gps = null_gp(X, Y, P)
        gps.optimize_all()
        return gps

    else:
        raise Exception('model not understood')


def run(X, Y, model1='se_cor', model2='null', P=2):
    """
    Input:
        - X of dim [N_samples, 2], numpy array, positions
        - Y of dim [N_samples, N_genes] expression levels
        - P number of genes to model jointly
        - model1 and model2: types of model to compare

    The two different models are trained and their LL is compared to get p values
    and q values

    Returns vectore of p values and q values for each gene or each gene-tuples
    """
    m1 = run_model(X, Y, model1, P)
    m2 = run_model(X, Y, model2, P)

    p_vals = util.pval(alt=m1, null=m2)
    q_vals = util.qvalue(p_vals)

    return p_vals, q_vals

if __name__ == '__main__':
    N = 500
    NG = 10
    X = np.reshape(np.random.randn(N*2),[N,2])
    Y = np.reshape(np.random.randn(N*NG),[N,NG])

    P = 2

    p_vals, q_vals = run(X, Y, P=P)
    print(( 'p_vals ', p_vals))
    print(( 'q_vals ', q_vals))
