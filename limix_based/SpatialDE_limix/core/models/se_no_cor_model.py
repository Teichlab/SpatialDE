import numpy as np
import scipy.special
import itertools

# form here
from base_model import SpatialGP
import SpatialDE_limix.core.utils.util as util

# limix objects
from limix.core.covar import SQExpCov, DiagonalCov
from limix.core.gp import GP2KronSum


class se_spatial_no_cor_gp(SpatialGP):
    """docstring for se_spatial_gp."""
    def __init__(self, X, Y, P=1):
        super(se_spatial_no_cor_gp, self).__init__(X, Y)
        self.P = P  # number of genes to consider jointly
        self.G = Y.shape[1]

        # indices of genes to test jointly and number of tests
        self.test_ix = [i for i in itertools.combinations(range(self.G),self.P)]
        self.N_test = int(scipy.special.binom(self.G, self.P))

        assert self.N_test == len(self.test_ix), "problem itertools"

        # results
        self.LML = np.array([np.Inf for i in range(self.N_test)])
        self.l = np.array([-1. for i in range(self.N_test)])
        self.parameters = [None for i in range(self.N_test)]

    def build_se(self, l):
        se = SQExpCov(self.X)
        se.length = l
        self.fixed_se = se.K()
        self.U, self.S = util.factor(self.fixed_se)

        # slower for some reason ...
        # se = util.SE_kernel(self.X,l)
        # self.U, self.S = util.factor(se)


    def build_limix_gp(self, Y):
        assert Y.shape[1] == self.P, 'dimension mismatch'

        Cg = DiagonalCov(self.P)
        Cn = DiagonalCov(self.P)

        return GP2KronSum(Y, Cg, Cn, S_R=self.S, U_R=self.U)

    def optimize_all(self, grid_size=10):
        l_grid = util.get_l_grid(self.X, grid_size)

        for l in l_grid:
            self.build_se(l)
            # for each gene or gene n-tuple (parallelised ? -> needs list of GPs then)
            for i in range(self.N_test):
                self.optimise_single(i, l)

    def optimise_single(self, i, l):
        Y_i = self.Y[:,self.test_ix[i]]
        gp = self.build_limix_gp(Y_i)
        gp.optimize(verbose=False)
        if gp.LML() < self.LML[i]:
            self.l[i] = l
            self.LML[i] = gp.LML()
            # TODO also store matrices to compute variance explained at the end
            # TODO also store parameters for interpretations ?


if __name__ == '__main__':
    N = 50
    NG = 10
    X = np.reshape(np.random.randn(N*2),[N,2])
    Y = np.reshape(np.random.randn(N*NG),[N,NG])

    P = 2

    gps = se_spatial_no_cor_gp(X, Y, P)
    gps.optimize_all()
    # print(gps.LML)
