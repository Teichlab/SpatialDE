import os
import glob
import warnings
import json
import logging

import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from matplotlib.image import imread

import h5py
from anndata import AnnData


def read_spaceranger(spaceranger_out_dir: str, read_images: bool = True) -> AnnData:
    """
    Read 10x SpaceRanger output.

    Args:
        spaceranger_out_dir: Path to the directory with SpaceRanger output.
        read_images: Whether to also read images into memory.

    Returns:
        An annotated data matrix.
    """
    fname = glob.glob(os.path.join(spaceranger_out_dir, "*filtered_feature_bc_matrix.h5"))
    if len(fname) == 0:
        raise FileNotFoundError(
            "filtered_feature_bc_matrix.h5 file not found in specified directory"
        )
    elif len(fname) > 1:
        warnings.warn(
            "Multiple files ending with filtered_feature_bc_matrix.h5 found in specified directory, using the first one",
            RuntimeWarning,
        )
    fname = fname[0]
    with h5py.File(fname, "r") as f:
        matrix = f["matrix"]
        sparsemat = csr_matrix(
            (matrix["data"][...], matrix["indices"][...], matrix["indptr"][...]),
            shape=matrix["shape"][...][::-1],
        )

        barcodes = matrix["barcodes"][...].astype(np.unicode)

        adata = AnnData(X=sparsemat)

        features = matrix["features"]
        adata.var_names = features["name"][...].astype(np.unicode)
        adata.var["id"] = features["id"][...].astype(np.unicode)
        for f in features["_all_tag_keys"]:
            feature = features[f][...]
            if feature.dtype.kind in ("S", "U"):
                feature = feature.astype(np.unicode)
            adata.var[f.astype(np.unicode)] = feature

    _, counts = np.unique(adata.var_names, return_counts=True)
    if np.sum(counts > 1) > 0:
        logging.warning("Duplicate gene names present. Converting to unique names.")
        adata.var_names_make_unique()

    tissue_positions = (
        pd.read_csv(
            os.path.join(spaceranger_out_dir, "spatial", "tissue_positions_list.csv"),
            names=(
                "barcode",
                "in_tissue",
                "array_row",
                "array_col",
                "pxl_col_in_fullres",
                "pxl_row_in_fullres",
            ),
        )
        .set_index("barcode")
        .loc[barcodes]
        .drop("in_tissue", axis=1)
    )
    adata.obsm["spatial"] = tissue_positions[
        ["pxl_row_in_fullres", "pxl_col_in_fullres"]
    ].to_numpy()
    adata.obs = tissue_positions.drop(["pxl_row_in_fullres", "pxl_col_in_fullres"], axis=1)

    with open(os.path.join(spaceranger_out_dir, "spatial", "scalefactors_json.json"), "r") as f:
        meta = json.load(f)
    adata.uns["spot_diameter_fullres"] = meta["spot_diameter_fullres"]
    if read_images:
        adata.uns["tissue_lowres_image"] = imread(
            os.path.join(spaceranger_out_dir, "spatial", "tissue_lowres_image.png")
        )
        adata.uns["tissue_hires_image"] = imread(
            os.path.join(spaceranger_out_dir, "spatial", "tissue_hires_image.png")
        )
        adata.uns["tissue_hires_scalef"] = meta["tissue_hires_scalef"]
        adata.uns["tissue_lowres_scalef"] = meta["tissue_lowres_scalef"]

    return adata
