# `FPFS`: Fourier Power Function Shaplets (A fast, accurate shear estimator)

Document: https://fpfs.readthedocs.io/en/latest/

## Installation

### Download
```shell
git clone https://github.com/mr-superonion/FPFS.git
```

### Install
```shell
cd FPFS
pip install ./
```

## Example for noiseless galaxies

```python
import fpfs
import numpy as np
import astropy.io.fits as pyfits

# Read PSF image
psfData=pyfits.getdata('data/psf_test.fits')
# Setup the FPFS task.
# For noiseless galaxies, no need to input
# model for noise power.
fpTask=fpfs.fpfsBase.fpfsTask(psfData)


# Read GAL image
galImgAll=pyfits.getdata('data/gal_test.fits')
# Put images into a list
imgList=[galImgAll[i*64:(i+1)*64,0:64] for i in range(4)]

# Measure FPFS moments
a=fpTask.measure(imgList)

# Measure FPFS ellipticity, FPFS response
# The weighting parameter
C=100
# Again, for noiseless galaxies, you do not need to set rev=True
# to revise second-order noise bias
b=fpfs.fpfsBase.fpfsM2E(a,C)
# Estimate shear
g_est=np.average(b['fpfs_e1'])/np.average(b['fpfs_RE'])
print('estimated shear is: %.5f' %g_est)
print('input shear is: 0.02')
```

## Reference
+ [version 2.0](https://ui.adsabs.harvard.edu/abs/2021arXiv211001214L/abstract)
+ [version 1.0](https://ui.adsabs.harvard.edu/abs/2018MNRAS.481.4445L/abstract)
