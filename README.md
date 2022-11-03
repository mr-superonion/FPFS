# `FPFS`: Fourier Power Function Shaplets (A fast, accurate shear estimator)
----
[![Python application](https://github.com/mr-superonion/FPFS/actions/workflows/python-app.yml/badge.svg?branch=master)](https://github.com/mr-superonion/FPFS/actions/workflows/python-app.yml)
[![Documentation Status](https://readthedocs.org/projects/fpfs/badge/?version=latest)](https://fpfs.readthedocs.io/en/latest/?badge=latest)


Fourier Power Function Shapelets (`FPFS`) is a perturbation-based estimator for
shear responses of galaxy shape, flux and detection --- It uses the
leading-order perturbations of shear (a vector perturbation) and image noise (a
tensor perturbation) to derive the shear responses and noise responses of
measurements and detections. It is a passive shear estimator --- It does not
repeatedly distort each observed galaxy to derive the shear responses; instead,
the shear responses are derived using the analytical shear responses of a set
of basis functions (Shapelets basis and peak basis).  This method can process
about 1000 galaxies in 1 cpu second, and it  has been tested with simple
simulations and demonstrated to control multiplicative shear estimation bias
below 1% even in the existence of blending.

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
pip install -e . --user
```
----

## Reference
The following papers are ready to be cited if you find any of these papers
interesting or use the pipeline. Comments are welcome.

+ **version 3:** [Li & Mandelbaum
  (2022)](https://ui.adsabs.harvard.edu/abs/2022arXiv220810522L/abstract)
  correct for detection bias from pixel level by interpreting smoothed pixel
  values as a projection of signal onto a set of basis functions.

+ **version 2:** [Li , Li & Massey
  (2022)](https://ui.adsabs.harvard.edu/abs/2021arXiv211001214L/abstract)
  derive the covariance matrix of FPFS measurements and corrects for noise bias
  to second-order. In addition, it derives the correction for selection bias.

+ **version 1:** [Li et. al
  (2018)](https://ui.adsabs.harvard.edu/abs/2018MNRAS.481.4445L/abstract)
  build up the FPFS formalism based on
  [Fourier_Quad](https://arxiv.org/abs/1312.5514) and [polar
  shapelets](https://arxiv.org/abs/astro-ph/0408445).
----
