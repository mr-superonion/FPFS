import numpy
from setuptools import setup, find_packages

setup(
    name='fpfs',
    version='2.0.2',
    description='FPFS shear estimator',
    author='Xiangchong Li et al.',
    author_email='mr.superonion@hotmail.com',
    python_requires='>=3.6',
    install_requires=[
        'numba',
        'numpy',
        'scipy',
        'galsim',
    ],
    include_dirs=numpy.get_include(),
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    url = "https://github.com/mr-superonion/FPFS/",
)
