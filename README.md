# `FPFS`: Fourier Power Function Shaplets (A fast, accurate shear estimator)
----

## Installation

For stable version:
```shell
pip install FPFS
```
Or clone the repository:
```shell
git clone https://github.com/mr-superonion/FPFS.git
cd FPFS
pip install -e .
```

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
+ [version 2.0](https://ui.adsabs.harvard.edu/abs/2021arXiv211001214L/abstract)
+ [version 1.0](https://ui.adsabs.harvard.edu/abs/2018MNRAS.481.4445L/abstract)

----
