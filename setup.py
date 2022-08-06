import os
from setuptools import setup, find_packages

# version of the package
__version__ = ''
fname = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "fpfs",
    "__version__.py")
with open(fname, 'r') as ff:
    exec(ff.read())

setup(
    name='fpfs',
    version=__version__,
    description='FPFS shear estimator',
    author='Xiangchong Li',
    author_email='mr.superonion@hotmail.com',
    python_requires='>=3.6',
    install_requires=[
        'numpy<1.23',
        'numba',
        'scipy',
        'galsim',
        'astropy',
        'matplotlib',
    ],
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    url = "https://github.com/mr-superonion/FPFS/",
)
