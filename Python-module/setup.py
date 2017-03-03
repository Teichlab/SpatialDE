from setuptools import setup, find_packages

setup(
        name='SpatialDE',
        version='0.2.0',
        description='Spatial and Temporal DE test',
        url='https://github.com/Teichlab/SpatialGP',
        packages=find_packages(),
        install_requires=['numpy', 'scipy', 'pandas', 'tqdm'],
        author='Valentine Svensson',
        author_email='valentine@nxn.se',
        license='MIT'
    )
