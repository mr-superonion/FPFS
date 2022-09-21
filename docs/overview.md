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

## References

The following papers is ready to be cited if you use the pipeline, or find them
interesting:

+ [version 3.0](https://ui.adsabs.harvard.edu/abs/2022arXiv220810522L/abstract)
This paper derives the correction for **detection bias**. Scripts used to produce
plots in the paper can be found [here](../notebooks/paper_2022).

+ [version 2.0](https://ui.adsabs.harvard.edu/abs/2021arXiv211001214L/abstract):
This paper derives the covariance matrix of FPFS measurements and corrects for
**noise bias** to second-order. In addition, it derives the correction for galaxy
sample **selection bias**. Scripts used to produce plots in the paper can be found
[here](../notebooks/paper_2021).

+ [version 1.0](https://ui.adsabs.harvard.edu/abs/2018MNRAS.481.4445L/abstract):
This paper builds up the FPFS formalism based on
[Fourier_Quad](https://arxiv.org/abs/1312.5514) and
[Shapelets](https://arxiv.org/abs/astro-ph/0408445).

## Developers

+ Xiangchong Li (xiangchl at andrew.cmu.edu)

If you have any trouble installing or using the code, or find a bug, or have a
suggestion for a new feature, please open up an Issue on our [GitHub
repository](https://github.com/mr-superonion/FPFS). We also accept pull
requests if you have something you’d like to contribute to the code base.
