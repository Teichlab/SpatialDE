import click
import numpy as np
import pandas as pd

import NaiveDE
import SpatialDE


@click.command()
@click.argument('expression_csv', type=click.Path(exists=True), metavar='<expression csv>')
@click.argument('coordinate_csv', type=click.Path(exists=True), metavar='<cooridnates csv>')
@click.argument('results_csv', type=click.Path(), metavar='<output file>')
@click.option('--model_selection_csv', type=click.Path(), default=None)
def main(expression_csv, coordinate_csv, results_csv, model_selection_csv):
    ''' Perform SpatialDE test on data in input files.

    <expression csv> : A CSV file with expression valies. Columns are genes,
    and Rows are samples

    <coordinates csv> : A CSV file with sample coordinates. Each row is a sample,
    the columns with coordinates must be named 'x' and 'y'. For other formats
    (e.g. 1d or 3d queries), it is recommended to write a custom Python
    script to do the analysis.

    <output file> : P-vaues and other relevant values for each gene
    will be stored in this file, in CSV format.

    '''
    df = pd.read_csv(expression_csv, index_col=0)

    df = df.T[df.sum(0) >= 3].T  # Filter practically unobserved genes

    sample_info = pd.read_csv(coordinate_csv, index_col=0)

    sample_info['total_counts'] = df.sum(1)
    sample_info = sample_info.query('total_counts > 5')  # Remove empty features

    df = df.loc[sample_info.index]
    X = sample_info[['x', 'y']]

    # Convert data to log-scale, and account for depth
    dfm = NaiveDE.stabilize(df.T).T
    res = NaiveDE.regress_out(sample_info, dfm.T, 'np.log(total_counts)').T

    # Perform Spatial DE test with default settings
    results = SpatialDE.run(X, res)

    # Save results and annotation in files for interactive plotting and interpretation
    results.to_csv(results_csv)

    if not model_selection_csv:
        return results

    de_results = results[(results.qval < 0.05)].copy()
    ms_results = SpatialDE.model_search(X, res, de_results)

    ms_results.to_csv(model_selection_csv)

    return results, ms_results

