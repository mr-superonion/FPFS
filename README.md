# `FPFS`: Fourier Power Function Shaplets (A fast, accurate shear estimator)
----
[![tests](https://github.com/mr-superonion/FPFS/actions/workflows/tests.yml/badge.svg?branch=master)](https://github.com/mr-superonion/FPFS/actions/workflows/tests.yml)
[![docs](https://readthedocs.org/projects/fpfs/badge/?version=latest)](https://fpfs.readthedocs.io/en/latest/?badge=latest)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


Fourier Power Function Shapelets (`FPFS`) is an innovative estimator for the
shear responses of galaxy shape, flux, and detection. Utilizing leading-order
perturbations of shear (a vector perturbation) and image noise (a tensor
perturbation), `FPFS` determines shear and noise responses for both
measurements and detections. Unlike traditional methods that distort each
observed galaxy repeatedly, `FPFS` employs analytical shear responses of select
basis functions, including Shapelets basis and peak basis. Remarkably
efficient, `FPFS` can process approximately 1,000 galaxies within a single CPU
second. Testing under simple simulations has proven its capability to maintain
a multiplicative shear estimation bias below 0.5%, even amidst blending
challenges. For further details, refer to the `FPFS` module documentation
[here](https://fpfs.readthedocs.io/en/latest/).

----

## Installation

For stable (old) version, which have not been updated:
```shell
pip install fpfs
```

Or clone the repository:
```shell
git clone https://github.com/mr-superonion/FPFS.git
cd FPFS
pip install -e . --user
```

Before using the code, please setup the jax environment
```shell
source fpfs_config
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

## Development

Before sending pull request, please make sure that the modified code passed the
pytest and flake8 tests. Run the following commands under the root directory
for the tests:

```shell
flake8
pytest -vv
```

----
