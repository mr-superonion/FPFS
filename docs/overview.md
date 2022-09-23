# Overview

FPFS is an open-source software for fast and accurate shear estimation.

## Basic installation

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

## Developers

+ Xiangchong Li (xiangchl at andrew.cmu.edu)

If you have any trouble installing or using the code, or find a bug, or have a
suggestion for a new feature, please open up an Issue on our [GitHub
repository](https://github.com/mr-superonion/FPFS). We also accept pull
requests if you have something you’d like to contribute to the code base.
