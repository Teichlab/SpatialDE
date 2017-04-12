import numpy as np

# form here
from base_model import SpatialGP
import ../utils/util as util

# limix objects
from limix.core.covar import SQExpCov, FreeFormCov
from limix.core.gp import GP2KronSum


class se_spatial_gp(SpatialGP):
    """docstring for se_spatial_gp."""
    def __init__(self, X, Y, grid_size = 10):
        super(se_spatial_gp, self).__init__(X, Y)
        self.l = 1.
        self.l_grid = util.get_l_grid(X, grid_size)

        self.build_covar()
        self.build_gp()

    def build_covar(self):
        self.build_se()
        self.Cg = FreeFormCov(self.P)
        self.Cn = FreeFormCov(self.P)

        # TODO implement smart initialisation (splitting empirical covariance)
        self.Cg.setRandomParams()
        self.Cn.setRandomParams()

    def build_se(self):
        self.se = SQExpCov(self.X)
        self.se.length = self.l
        self.fixed_se = self.se.K()

    def build_gp(self):
        self.gp = GP2KronSum(self.Y, self.Cg, self.Cn, R=self.fixed_se)

    def optimize(self):
        best_l = self.l_grid[0]
        best_params = self.gp.getParams().copy()
        best_LML = np.Inf

        for l in self.l_grid:
            self.l = l
            # NOTE other covariance terms are not changed -> parameters from previous optimisation
            # are kept
            self.build_se()
            self.build_gp()
            self.gp.optimize()

            # for each gene or gene pair or gene n-tuple (parallelised ? -> needs list of GPs then)

            if self.gp.LML() < best_LML:
                best_l = l
                best_params = self.gp.getParams().copy()
                best_LML = self.gp.LML()

        self.l = best_l
        self.build_se()
        self.build_gp()
        self.gp.setParams(best_params)


if __name__ == '__main__':
    N = 100
    X = np.reshape(np.random.randn(N*2),[N,2])
    Y = np.reshape(np.random.randn(N*2),[N,2])

    gp = se_spatial_gp(X, Y)
    gp.optimize()
    print gp.l, gp.gp.getParams()
