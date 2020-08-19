from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent

setup(
    name='SpatialDE',
    version='1.1.3',
    description='Spatial and Temporal DE test',
    long_description=(HERE.parent / 'README.rst').read_text(),
    url='https://github.com/Teichlab/SpatialDE',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy', 'scipy >= 1.0', 'pandas>=0.23', 'tqdm',
        'NaiveDE', 'Click',
    ],
    entry_points=dict(
        console_scripts=['spatialde=SpatialDE.scripts.spatialde_cli:main'],
    ),
    author='Valentine Svensson',
    author_email='valentine@nxn.se',
    license='MIT',
)
