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
        y = np.random.multivariate_normal(0 * X[:, 0], K)

        plt.subplot(1, len(ls_list), i + 1)
        plt.pcolormesh(X1, X2, y.reshape(X1.shape), cmap=cm.inferno)
        plt.contour(X1, X2, y.reshape(X1.shape), cmap=cm.inferno)
        plt.axis('equal')
        plt.title('$ \ell = {} $'.format(ls))


    fig = plt.gcf()
    fig.set_size_inches(9, 3)
    plt.savefig('ls_guide.png')


if __name__ == '__main__':
    ls_sample_2d([1., 5., 10., 20.])
