# `FPFS`: Fourier Power Function Shaplets (A fast, accurate shear estimator)
----
[![Python application](https://github.com/mr-superonion/FPFS/actions/workflows/python-app.yml/badge.svg?branch=master)](https://github.com/mr-superonion/FPFS/actions/workflows/python-app.yml)
[![Documentation Status](https://readthedocs.org/projects/fpfs/badge/?version=latest)](https://fpfs.readthedocs.io/en/latest/?badge=latest)

`FPFS` is a perturbation-based shear estimator: It uses the leading order
perturbations of shear (which is a vector perturbation) and noise (which is a
tensor perturbation) to construct shear estimator and revise noise bias,
respectively. It is a passive shear estimator: We do not repeatedly distort each
observed galaxy to obtain the responses of galaxy properties to shear
distortion; instead, the responses are derived using the analytical shear
responses of a set of basis functions.

Documentation for FPFS modules can be found [here](https://fpfs.readthedocs.io/en/latest/)

----

## Installation

For stable version:
```shell
pip install fpfs
```

Or clone the repository:
```shell
git clone https://github.com/mr-superonion/FPFS.git
cd FPFS
pip install -e .
```
----

## Reference
+ [version 3.0](https://arxiv.org/abs/2208.10522):
This paper correct for detection bias by interpreting smoothed pixel values as
a projection of signal onto a set of basis functions.

+ [version 2.0](https://ui.adsabs.harvard.edu/abs/2021arXiv211001214L/abstract):
This paper derives the covariance matrix of FPFS measurements and corrects for
noise bias to second-order. In addition, it derives the correction for
selection bias (including Kaiser flow and ellipticity-flux measurement error
correlation).
+ [version 1.0](https://ui.adsabs.harvard.edu/abs/2018MNRAS.481.4445L/abstract):
This paper builds up the FPFS formalism based on
[Fourier_Quad](https://arxiv.org/abs/1312.5514) and
[Shapelets](https://arxiv.org/abs/astro-ph/0408445).
----
