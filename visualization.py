import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

fgp = __import__('FaST-GP')

def ls_sample_2d(ls_list=[3, 10, 30, 100], xmin=5, xmax=30, ymin=5, ymax=26):
    x = np.linspace(xmin, xmax)
    y = np.linspace(ymin, ymax)

    X1, X2 = np.meshgrid(x, y)
    X = np.vstack((X1.flatten(), X2.flatten())).T

    for i, ls in enumerate(ls_list):
        K = fgp.SE_kernel(X, ls)
        Y = np.random.multivariate_normal(0 * X[:, 0], K)

        plt.subplot(1, len(ls_list), i + 1)
        plt.pcolormesh(X1, X2, Y.reshape(X1.shape), cmap=cm.inferno)
        plt.contour(X1, X2, Y.reshape(X1.shape), cmap=cm.inferno)
        plt.axis('equal')
        plt.title('$ \ell = {} $'.format(ls))


    fig = plt.gcf()
    fig.set_size_inches(9, 3)
    plt.savefig('ls_guide.png')


def linear_sample_2d(xmin=5, xmax=30, ymin=5, ymax=26):
    x = np.linspace(xmin, xmax)
    y = np.linspace(ymin, ymax)

    X1, X2 = np.meshgrid(x, y)
    X = np.vstack((X1.flatten(), X2.flatten())).T

    K = fgp.linear_kernel(X)
    Y = np.random.multivariate_normal(0 * X[:, 0], K)

    plt.pcolormesh(X1, X2, Y.reshape(X1.shape), cmap=cm.inferno)
    plt.contour(X1, X2, Y.reshape(X1.shape), cmap=cm.inferno)
    plt.axis('equal')

    fig = plt.gcf()
    fig.set_size_inches(9, 3)
    plt.savefig('linear_guide.png')


def period_sample_2d(p_list=[3, 10, 30, 100], xmin=5, xmax=30, ymin=5, ymax=26):
    x = np.linspace(xmin, xmax)
    y = np.linspace(ymin, ymax)

    X1, X2 = np.meshgrid(x, y)
    X = np.vstack((X1.flatten(), X2.flatten())).T

    for i, p in enumerate(p_list):
        K = fgp.cosine_kernel(X, p)
        Y = np.random.multivariate_normal(0 * X[:, 0], K)

        plt.subplot(1, len(p_list), i + 1)
        plt.pcolormesh(X1, X2, Y.reshape(X1.shape), cmap=cm.inferno)
        plt.contour(X1, X2, Y.reshape(X1.shape), cmap=cm.inferno)
        plt.axis('equal')
        plt.title('$ p = {} $'.format(p))


    fig = plt.gcf()
    fig.set_size_inches(9, 3)
    plt.savefig('p_guide.png')


if __name__ == '__main__':
    # ls_sample_2d([1., 5., 10., 20.])
    # linear_sample_2d()
    period_sample_2d([1., 5., 10., 20.])
