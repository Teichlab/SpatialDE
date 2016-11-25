import click
import numpy as np
import pandas as pd
import GPflow
from tqdm import tqdm

@click.command()
@click.argument('expression_csv')
@click.argument('results_csv')
def fit_gps(expression_csv, results_csv):
    df = pd.read_csv(expression_csv, index_col=0)
    coords = pd.DataFrame(index=df.index)
    coords['x'] = df.index.str.split('x').str.get(0).map(float)
    coords['y'] = df.index.str.split('x').str.get(1).map(float)

    X = coords[['x', 'y']].as_matrix()
    Y = np.log10(df.iloc[:, 0].map(float)[:, None] + 1)

    k = GPflow.kernels.RBF(2, ARD=False) + GPflow.kernels.Constant(2)
    m = GPflow.gpr.GPR(X, Y, k)
    m_init = m.get_free_state() * 0 + 1.

    k_flat = GPflow.kernels.Constant(2)
    m_flat = GPflow.gpr.GPR(X, Y, k_flat)
    m_flat_init = m_flat.get_free_state() * 0 + 1.

    results = pd.DataFrame(index=df.columns)
    results['lengthscale'] = np.nan
    results['rbf_ll'] = np.nan
    results['constant_ll'] = np.nan

    for g in tqdm(df.columns):
        m.Y = np.log10(df[g].map(float)[:, None] + 1)
        m.set_state(m_init)
        o = m.optimize()

        m_flat.Y = m.Y.value
        m_flat.set_state(m_flat_init)
        o_flat = m_flat.optimize()

        results.loc[g, 'lengthscale'] = m.get_parameter_dict()['model.kern.rbf.lengthscales'][0]
        results.loc[g, 'rbf_ll'] = -o.fun
        results.loc[g, 'constant_ll'] = -o_flat.fun

    results.to_csv(results_csv)

if __name__ == '__main__':
    fit_gps()
