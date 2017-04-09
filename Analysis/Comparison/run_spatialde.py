from time import time

import click
import numpy as np
import pandas as pd

import SpatialDE


@click.command()
@click.argument('prefix')
def run_spatialde(prefix):
    X = pd.read_csv(prefix + '_X.csv', index_col=0)
    dfm = pd.read_csv(prefix + '_expression.csv', index_col=0)
    t0 = time()
    SpatialDE.run(X, dfm)
    total_time = time() - t0

    pd.DataFrame({'time': total_time}, index=[prefix]) \
        .to_csv(prefix + '_spatialde_time.csv')


if __name__ == '__main__':
    run_spatialde()
