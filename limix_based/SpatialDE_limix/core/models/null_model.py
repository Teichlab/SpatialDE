import numpy as np
import scipy.special
import itertools

# form here
from base_model import SpatialGP
import SpatialDE_limix.core.utils.util as util

# limix objects
from limix.core.covar import SQExpCov, FreeFormCov
from limix.core.gp import GP2KronSum


class null_gp(SpatialGP):
    """
    input:
        - X of dimensions [N_sample, 2] -> positions
        - Y of dim [N_sample, N_gene] -> gene expression
        - P the number of genes to model jointly

    Model:
        - Gene P-tuples are modeled with a GP with covariance FreeForm * identity noise
        This means genes are not independent, but there is no spatial DE or correlation

    The LML is computed for each gene tuples and stored into self.LML
    parameters of the model not stored yet

    """
    def __init__(self, X, Y, P=1):
        super(null_gp, self).__init__(X, Y)
        self.P = P  # number of genes to consider jointly
        self.G = Y.shape[1]
        self.N_params = FreeFormCov(self.P).getNumberParams()

        # indices of genes to test jointly and number of tests
        self.test_ix = [i for i in itertools.combinations(range(self.G),self.P)]
        self.N_test = int(scipy.special.binom(self.G, self.P))

        # results
        self.LML = np.array([np.Inf for i in range(self.N_test)])
        self.parameters = [None for i in range(self.N_test)]

    def build_limix_gp(self, Y):
        assert Y.shape[1] == self.P, 'dimension mismatch'

        Cg = FreeFormCov(self.P)
        Cn = FreeFormCov(self.P)
        R = np.eye(self.N)

        # Cg is initialised with zeros and will stay as such
        Cn.setRandomParams()

        return GP2KronSum(Y, Cg, Cn, R=R)

    def optimize_all(self):
        for i in xrange(self.N_test):
            self.optimise_single(i)

    def optimise_single(self, i):
        Y_i = self.Y[:,self.test_ix[i]]
        gp = self.build_limix_gp(Y_i)
        gp.optimize(verbose=False)
        self.LML[i] = gp.LML()


if __name__ == '__main__':
    N = 500
    NG = 10
    X = np.reshape(np.random.randn(N*2),[N,2])
    Y = np.reshape(np.random.randn(N*NG),[N,NG])

    P = 2

    gps = null_gp(X, Y, P)
    gps.optimize_all()
    # print(gps.LML)
