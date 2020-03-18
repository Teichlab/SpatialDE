import numpy as np
import pandas as pd
import tensorflow as tf

from enum import Enum, auto
from typing import Optional, Union

import logging

np_float = Union[np.float16, np.float32, np.float64, np.float128]
float = Union[float, np_float]

class Kernel(Enum):
    null = auto()
    const = auto()
    linear = auto()
    SE = auto()
    PER = auto()

class GP(Enum):
    GPR = auto()
    SGPR  = auto()

class SGPIPM(Enum):
    free = auto()
    random = auto()
    grid = auto()

class GPControl:
    def __init__(self, gp: Optional[GP] = None, inducing_point_method: SGPIPM = SGPIPM.grid, n_kernel_components : Optional[int] = 5, ard : Optional[bool] = True, n_inducers: Optional[int] = None):
        self.gp = gp
        self.ipm = inducing_point_method
        self.ncomponents = n_kernel_components
        self.ard = ard
        self.ninducers = n_inducers

def get_dtype(df: pd.DataFrame, msg="Data frame"):
    dtys = df.dtypes.unique()
    if dtys.size > 1:
        logging.warning("%s has more than one dtype, selecting the first one" % msg)
    return dtys[0]

def bh_adjust(pvals):
    order = np.argsort(pvals)
    alpha = np.minimum(1, np.maximum.accumulate(len(pvals) / np.arange(1, len(pvals) + 1) * pvals[order]))
    return alpha[np.argsort(order)]
