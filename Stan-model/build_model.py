import pickle

import click
import pystan


@click.command()
@click.argument('file_path')
def build_model(file_path):
    stan_model = pystan.StanModel(file=file_path)

    out_file_path = file_path.replace('.stan', '.pkl')
    with open(out_file_path, 'wb') as fh:
        pickle.dump(stan_model, fh)


if __name__ == '__main__':
    build_model()
