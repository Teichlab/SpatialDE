import logging
from time import time
import warnings
from itertools import zip_longest
from typing import Optional, Dict, Tuple, Union, List

import numpy as np
import pandas as pd

from anndata import AnnData

from tqdm.auto import tqdm

from ._internal.util import DistanceCache
from ._internal.util import bh_adjust, calc_sizefactors, default_kernel_space, kspace_walk
from ._internal.score_test import (
    NegativeBinomialScoreTest,
    combine_pvalues,
)
from ._internal.tf_dataset import AnnDataDataset

def _add_individual_score_test_result(resultdict, kernel, kname, gene):
    if "kernel" not in resultdict:
        resultdict["kernel"] = [kname]
    else:
        resultdict["kernel"].append(kname)
    if "gene" not in resultdict:
        resultdict["gene"] = [gene]
    else:
        resultdict["gene"].append(gene)
    for key, var in vars(kernel).items():
        if key[0] != "_":
            if key not in resultdict:
                resultdict[key] = [var]
            else:
                resultdict[key].append(var)
    return resultdict


def test(
    adata: AnnData,
    omnibus: bool = False,
    spatial_key="spatial",
    kernel_space: Optional[Dict[str, Union[float, List[float]]]] = None,
    sizefactors: Optional[np.ndarray] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logging.info("Performing DE test")

    X = adata.obsm[spatial_key]
    dcache = DistanceCache(X)
    if sizefactors is None:
        sizefactors = calc_sizefactors(adata)
    if kernel_space is None:
        kernel_space = default_kernel_space(X, dcache)

    individual_results = None if omnibus else []
    if adata.n_obs <= 2000 or omnibus:
        kernels = []
        kernelnames = []
        for k, name in kspace_walk(kernel_space, dcache):
            kernels.append(k)
            kernelnames.append(name)
        test = NegativeBinomialScoreTest(
            sizefactors,
            omnibus,
            kernels,
        )

        results = []
        with tqdm(total=adata.n_vars) as pbar:
            for i, (y, g) in AnnDataDataset(adata, dtype=test.dtype).enumerate():
                i = i.numpy()
                g = g.numpy().decode("utf-8")
                pbar.update()
                t0 = time()
                result, _ = test(y)
                t = time() - t0
                res = {"gene": g, "time": t}
                resultdict = result.to_dict()
                if omnibus:
                    res.update(resultdict)
                else:
                    res["pval"] = combine_pvalues(result).numpy()
                results.append(res)
                if not omnibus:
                    for k, n in zip(kernels, kernelnames):
                        _add_individual_score_test_result(resultdict, k, n, g)
                    individual_results.append(resultdict)
    else:  # doing all tests at once with stacked kernels leads to excessive memory consumption
        results = [[0, []] for _ in range(adata.n_vars)]
        nullmodels = []
        test = NegativeBinomialScoreTest(sizefactors)
        for k, n in kspace_walk(kernel_space, dcache):
            test.kernel = k
            if len(nullmodels) > 0:
                nullit = nullmodels
                havenull = True
            else:
                nullit = ()
                havenull = False
            with tqdm(total=adata.n_vars) as pbar:
                for null, (i, (y, g)) in zip_longest(nullit, AnnDataDataset(adata, dtype=test.dtype).enumerate()):
                    i = i.numpy()
                    g = g.numpy().decode("utf-8")

                    t0 = time()
                    res, null = test(y, null)
                    t = time() - t0
                    if not havenull:
                        nullmodels.append(null)
                    pbar.update()
                    results[i][0] += t
                    results[i][1].append(res)
                    resultdict = res.to_dict()
                    individual_results.append(_add_individual_score_test_result(resultdict, k, n, g))
        for i, g in enumerate(adata.var_names):
            results[i] = {
                "gene": g,
                "time": results[i][0],
                "pval": combine_pvalues(results[i][1]).numpy(),
            }

    results = pd.DataFrame(results)
    results["p.adj"] = bh_adjust(results.pval.to_numpy())

    if individual_results is not None:
        merged = {}
        for res in individual_results:
            for k, v in res.items():
                if k not in merged:
                    merged[k] = v if not np.isscalar(v) else [v]
                else:
                    if isinstance(merged[k], np.ndarray):
                        merged[k] = np.concatenate((merged[k], v))
                    elif isinstance(v, list):
                        merged[k].extend(v)
                    else:
                        merged[k].append(v)
        individual_results = pd.DataFrame(merged)
    return results, individual_results
