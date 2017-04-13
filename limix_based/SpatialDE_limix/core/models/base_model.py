

class SpatialGP(object):
    """
    Parent class of all spatial gp models
    """
    # TODO should Y include all genes or just noe at a time
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        self.N = self.Y.shape[0]

        assert self.X.shape[0] == self.N, 'dimension missmatch'
