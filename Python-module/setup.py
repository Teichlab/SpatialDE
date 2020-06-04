from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent

setup(
    name="SpatialDE",
    version="2.0.0-dev",
    description="Spatial and Temporal DE test",
    long_description=(HERE / "README.rst").read_text(),
    url="https://github.com/Teichlab/SpatialDE",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy >= 1.0",
        "pandas >= 1.0",
        "matplotlib >= 3.0",
        "tqdm",
        "Click",
        "gpflow >= 2.0",
        "anndata >= 0.7",
        "NaiveDE",
        "h5py",
    ],
    entry_points=dict(console_scripts=["spatialde=SpatialDE.scripts.spatialde_cli:main"],),
    author="Valentine Svensson",
    author_email="valentine@nxn.se",
    license="MIT",
)
