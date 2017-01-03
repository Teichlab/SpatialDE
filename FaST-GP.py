import numpy as np
from scipy.spatial.distance import pdist, squareform


def SE_kernel(X, l):
    R = squareform(pdist(X, 'euclidean')) ** 2
    return np.exp(R / (2 * l ** 2))


def factor(K):
    S, U = np.linalg.eigh(K)
    # .clip removes negative eigenvalues
    return U, S.clip(0.)


def get_UT1(U):
    return U.T.sum(0)[:, None]


def get_UTy(U, y):
    return U.T.dot(y)


def mu_hat(delta, UTy, UT1, S, n):
    ''' ML Estimate of bias mu, function of delta.
    '''
    Sd = (S + delta)[:, None]

    sum_1 = (UT1 / Sd).T.dot(UT1)[0]
    sum_2 = (UT1 / Sd).T.dot(UTy)[0]

    return sum_2 / sum_1


def LL(delta, UTy, UT1, S, n):
    ''' Log-likelihood of GP model as a function of delta.
    '''
    mu_h = mu_hat(delta, UTy, UT1, S, n)
    Sd = (S + delta)[:, None]
    sum_1 = np.log(Sd).sum()
    sum_2 = ((UTy - UT1 * mu_h) / Sd).sum()

    return -0.5 * (n * np.log(2 * np.pi) + sum_1 + n + n * np.log(sum_2))
