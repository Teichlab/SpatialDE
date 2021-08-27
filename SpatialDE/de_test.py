import logging
from time import time
import warnings
from itertools import zip_longest
from typing import Optional, Dict, Tuple, Union, List

import numpy as np
import pandas as pd
from gpflow import default_float

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


def _merge_individual_results(individual_results):
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
    return pd.DataFrame(merged)


def test(
    adata: AnnData,
    layer: Optional[str] = None,
    omnibus: bool = False,
    spatial_key: str = "spatial",
    kernel_space: Optional[Dict[str, Union[float, List[float]]]] = None,
    sizefactors: Optional[np.ndarray] = None,
    stack_kernels: Optional[bool] = None,
    use_cache: bool = True,
) -> Tuple[pd.DataFrame, Union[pd.DataFrame, None]]:
    """
    Test for spatially variable genes.

    Perform a score test to detect spatially variable genes in a spatial transcriptomics
    dataset. Multiple kernels can be tested to detect genes with different spatial patterns and lengthscales.
    The test uses a count-based likelihood and thus operates on raw count data. Two ways of handling multiple
    kernels are implemented: omnibus and Cauchy combination. The Cauchy combination tests each kernel separately
    and combines the p-values afterwards, while the omnibus test tests all kernels simultaneously. With multiple
    kernels the omnibus test is faster, but may have slightly less statistical power than the Cauchy combination.

    Args:
        adata: The annotated data matrix.
        layer: Name of the AnnData object layer to use. By default ``adata.X`` is used.
        omnibus: Whether to do an omnibus test.
        spatial_key: Key in ``adata.obsm`` where the spatial coordinates are stored.
        kernel_space: Kernels to test against. Dictionary with the name of the kernel function as key and list of
            lengthscales (if applicable) as values. Currently, three kernel functions are known:

            * ``SE``, the squared exponential kernel :math:`k(\\boldsymbol{x}^{(1)}, \\boldsymbol{x}^{(2)}; l) = \\exp\\left(-\\frac{\\lVert \\boldsymbol{x}^{(1)} - \\boldsymbol{x}^{(2)} \\rVert}{l^2}\\right)`
            * ``PER``, the periodic kernel :math:`k(\\boldsymbol{x}^{(1)}, \\boldsymbol{x}^{(2)}; l) = \\cos\\left(2 \pi \\frac{\\sum_i (x^{(1)}_i - x^{(2)}_i)}{l}\\right)`
            * ``linear``, the linear kernel :math:`k(\\boldsymbol{x}^{(1)}, \\boldsymbol{x}^{(2)}) = (\\boldsymbol{x}^{(1)})^\\top \\boldsymbol{x}^{(2)}`

            By default, 5 squared exponential and 5 periodic kernels with lengthscales spanning the range of the
            data will be used.
        sizefactors: Scaling factors for the observations. Default to total read counts.
        stack_kernels: When using the Cauchy combination, all tests can be performed in one operation by stacking
            the kernel matrices. This leads to increased memory consumption, but will drastically improve runtime
            on GPUs for smaller data sets. Defaults to ``True`` for datasets with less than 2000 observations and
            ``False`` otherwise.
        use_cache: Whether to use a pre-computed distance matrix for all kernels instead of computing the distance
            matrix anew for each kernel. Increases memory consumption, but is somewhat faster.

    Returns:
        If ``omnibus==True``, a tuple with a Pandas DataFrame as the first element and ``None`` as the second.
        The DataFrame contains the results of the test for each gene, in particular p-values and BH-adjusted p-values.
        Otherwise, a tuple of two DataFrames. The first contains the combined results, while the second contains results
        from individual tests.
    """
    logging.info("Performing DE test")

    X = adata.obsm[spatial_key]
    dcache = DistanceCache(X, use_cache)
    if sizefactors is None:
        sizefactors = calc_sizefactors(adata)
    if kernel_space is None:
        kernel_space = default_kernel_space(dcache)

    individual_results = None if omnibus else []
    if stack_kernels is None and adata.n_obs <= 2000 or stack_kernels or omnibus:
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
            for i, (y, g) in AnnDataDataset(adata, dtype=default_float(), layer=layer).enumerate():
                i = i.numpy()
                g = g.numpy().decode("utf-8")
                t0 = time()
                result, _ = test(y)
                t = time() - t0
                pbar.update()
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
                for null, (i, (y, g)) in zip_longest(
                    nullit, AnnDataDataset(adata, dtype=default_float(), layer=layer).enumerate()
                ):
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
                    individual_results.append(
                        _add_individual_score_test_result(resultdict, k, n, g)
                    )
        for i, g in enumerate(adata.var_names):
            results[i] = {
                "gene": g,
                "time": results[i][0],
                "pval": combine_pvalues(results[i][1]).numpy(),
            }

    results = pd.DataFrame(results)
    results["padj"] = bh_adjust(results.pval.to_numpy())

    if individual_results is not None:
        individual_results = _merge_individual_results(individual_results)
    return results, individual_results
