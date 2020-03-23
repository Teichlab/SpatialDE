import numpy as np

def bh_adjust(pvals):
    order = np.argsort(pvals)
    alpha = np.minimum(1, np.maximum.accumulate(len(pvals) / np.arange(1, len(pvals) + 1) * pvals[order]))
    return alpha[np.argsort(order)]
