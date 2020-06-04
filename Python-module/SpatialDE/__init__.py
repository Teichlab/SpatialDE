from .de_test import test
#from .aeh import fit_patterns
#from .aeh import spatial_patterns
from .gaussian_process import  GP, GPControl, fit, fit_fast, fit_detailed
from .dp_hmrf import (
    tissue_segmentation,
    TissueSegmentationParameters,
    TissueSegmentationStatus,
    TissueSegmentation,
)

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

