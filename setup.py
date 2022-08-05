from setuptools import setup, find_packages

setup(
    name='fpfs',
    version='3.0.1',
    description='FPFS shear estimator',
    author='Xiangchong Li',
    author_email='mr.superonion@hotmail.com',
    python_requires='>=3.6',
    install_requires=[
        'numba',
        'scipy',
        'galsim',
        'fitsio',
        'matplotlib',
    ],
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    url = "https://github.com/mr-superonion/FPFS/",
)
