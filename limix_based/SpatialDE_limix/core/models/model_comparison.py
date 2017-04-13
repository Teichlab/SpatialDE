import numpy as np

# from here
from SpatialDE_limix.core.models import null_gp, se_spatial_no_cor_gp, se_spatial_gp
from SpatialDE_limix.core.utils import util


def run_model(X, Y, model, P):
    if model == 'se_cor':
        gps = se_spatial_gp(X, Y, P)
        gps.optimize_all()
        return gps

    if model == 'se_no_cor':
        gps = se_spatial_no_cor_gp(X, Y, P)
        gps.optimize_all()
        return gps

    if model == 'null':
        gps = null_gp(X, Y, P)
        gps.optimize_all()
        return gps


def run(X, Y, model1='se_cor', model2='null', P=2):
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
    print 'p_vals ', p_vals
    print 'q_vals ', q_vals
