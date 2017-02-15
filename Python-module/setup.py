from setuptools import setup, find_packages

setup(
        name='SpatialDE',
        version='0.2.0',
        description='Spatial and Temporal DE test',
        packages=find_packages(),
        install_requires=['numpy', 'pandas', 'tqdm'],
        author='Valentine Svensson',
        author_email='valentine@nxn.se',
        license='MIT'
    )
