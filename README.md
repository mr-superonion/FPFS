# `FPFS`: Fourier Power Function Shaplets (A fast, accurate shear estimator)
----
[![Python application](https://github.com/mr-superonion/FPFS/actions/workflows/python-app.yml/badge.svg?branch=master)](https://github.com/mr-superonion/FPFS/actions/workflows/python-app.yml)

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
Documentation for FPFS modules can be found [here](https://fpfs.readthedocs.io/en/latest/)
----

## Demos

### Isolated galaxies
[demo1](https://github.com/mr-superonion/FPFS/blob/master/notebook/demos/demo1.ipynb)
simulates noisy galaxies with different SNRs and processes the galaxies with FPFS
shear estimator.
+   fpfs.simutil.sim_test: a wrapper of galsim to simulate galaxies for simple tests;
+   fpfs.fpfsBase.fpfsTask: a task to process galaxy images and measure shear.

[demo2](https://github.com/mr-superonion/FPFS/blob/master/notebook/demos/demo2.ipynb)
estimates shear under a more realistic situation with *PSF errors*.

### blended galaxies
TBD

----

## Reference
+ [version 2.0](https://ui.adsabs.harvard.edu/abs/2021arXiv211001214L/abstract):
This paper derives the covariance matrix of FPFS measurements and corrects for
noise bias to second-order. In addition, it derives the correction for
selection bias (including Kaiser flow and ellipticity-flux measurement error
correlation). Scripts used to produce plots in the paper can be found
[here](https://github.com/mr-superonion/FPFS/tree/master/notebook/paper-FPFS2021).
+ [version 1.0](https://ui.adsabs.harvard.edu/abs/2018MNRAS.481.4445L/abstract):
This paper builds up the FPFS formalism based on
[Fourier_Quad](https://arxiv.org/abs/1312.5514) and
[Shapelets](https://arxiv.org/abs/astro-ph/0408445).
----
