import pickle
from time import time

import click
import numpy as np
import pandas as pd
from tqdm import tqdm


@click.command()
@click.argument('prefix')
def run_stan(prefix):
    X = pd.read_csv(prefix + '_X.csv', index_col=0)
    dfm = pd.read_csv(prefix + '_expression.csv', index_col=0)

    with open('../../Stan-model/AltModel.pkl', 'rb') as fh:
        model = pickle.load(fh)

    t0 = time()
    for gene in tqdm(dfm.sample(n=100, axis=1)):
        data = {
            'N': X.shape[0],
            'D': 1,
            'X': X,
            'Y': dfm[gene]
        }
        model.optimizing(data=data)

    total_time = time() - t0

    pd.DataFrame({'time': total_time}, index=[prefix]) \
        .to_csv(prefix + '_stan_time.csv')


if __name__ == '__main__':
    run_stan()
