import scipy as sp
from scipy import interpolate


def qvalue(pv, pi0=None):
    """
    Estimates q-values from p-values.

    Args:
        pv (ndarray): Array of p-values.
        pi0 (float): If None, it's estimated as suggested in Storey and Tibshirani, 2003.

    Returns:
        ndarray: Array of q-values with the same shape as pv.

    Raises:
        ValueError: If pv is not a NumPy array or has invalid values.
        ValueError: If pi0 is not between 0 and 1.

    """
    if not isinstance(pv, sp.ndarray):
        raise ValueError("pv should be a NumPy array")

    if not (pv.min() >= 0 and pv.max() <= 1):
        raise ValueError("p-values should be between 0 and 1")

    original_shape = pv.shape
    pv = pv.ravel()

    m = float(len(pv))

    if len(pv) < 100 and pi0 is None:
        pi0 = 1.0
    elif pi0 is not None:
        pi0 = pi0
    else:
        counts = sp.array([(pv > i).sum() for i in sp.arange(0, 0.9, 0.01)])
        lam = sp.arange(0, 0.90, 0.01)
        pi0 = counts / (m * (1 - lam))

        tck = interpolate.splrep(lam, pi0, k=3)
        pi0 = interpolate.splev(lam[-1], tck)

        if pi0 > 1:
            pi0 = 1.0

    if not (0 <= pi0 <= 1):
        raise ValueError("pi0 is not between 0 and 1: %f" % pi0)

    p_ordered = sp.argsort(pv)
    pv = pv[p_ordered]
    qv = pi0 * m / len(pv) * pv
    qv[-1] = min(qv[-1], 1.0)

    i = sp.arange(len(pv) - 2, -1, -1)
    qv[i] = sp.minimum(pi0 * m * pv[i] / (i + 1.0), qv[i + 1])

    qv_temp = qv.copy()
    qv = sp.zeros_like(qv)
    qv[p_ordered] = qv_temp

    qv = qv.reshape(original_shape)

    return qv
