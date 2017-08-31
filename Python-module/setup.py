from setuptools import setup, find_packages

setup(
        name='SpatialDE',
        version='0.4.0',
        description='Spatial and Temporal DE test',
        url='https://github.com/Teichlab/SpatialDE',
        packages=find_packages(),
        include_package_data=True,
        install_requires=['numpy', 'scipy', 'pandas', 'tqdm',
                          'NaiveDE', 'Click'],
        entry_points='''
            [console_scripts]
            spatialde=SpatialDE.scripts.spatialde_cli:main
        ''',
        author='Valentine Svensson',
        author_email='valentine@nxn.se',
        license='MIT'
    )
