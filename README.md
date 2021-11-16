# `FPFS`: Fourier Power Function Shaplets (A fast, accurate shear estimator)

## Install

```shell
git clone https://github.com/mr-superonion/FPFS.git
pip install ./
```

## optional dependency

### Galsim
Galsim package is used for galaxy image simulation
to test the `FPFS` shear estimator.

```shell
pip install galsim
```
or
```shell
python ./setup.py install
```

## Example for a noiseless galaxy

```python
import fpfs
import astropy.io.fits as pyfits

# Read PSF image
psfData=pyfits.getdata('data/psf_test.fits')
# Setup the task
fpTask=fpfs.fpfsBase.fpfsTask(psfData)


# Read GAL image
galImgAll=pyfits.getdata('data/gal_test.fits')
imgList=[galImgAll[i*64:(i+1)*64,0:64] for i in range(4)]

# Measure FPFS moments
a=fpTask.measure(imgList)

# Measure FPFS ellipticity, FPFS response
# The weighting parameter
C=100
# You can set rev=True to revise second-order noise bias
b=fpfs.fpfsBase.fpfsM2E(a,C)
# Estimate shear
g_est=np.average(b['fpfs_e1'])/np.average(b['fpfs_RE'])
print('estimated shear is: %.5f' %g_est)
print('input shear is: 0.02')
```

## Reference
+ [version 2.0](https://ui.adsabs.harvard.edu/abs/2021arXiv211001214L/abstract)
+ [version 1.0](https://ui.adsabs.harvard.edu/abs/2018MNRAS.481.4445L/abstract)
