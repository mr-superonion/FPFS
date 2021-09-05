# `FPFS`: Fourier Power Function Shaplets (A simple, accurate shear estimator)

## Install

```shell
pip install ./
```

## optional dependency

### Galsim
Galsim package is used for galaxy image simulation
to test the `FPFS` shear estimator.
```shell
pip install galsim
```

## Example

```python
import fpfs
import astropy.io.fits as pyfits

# read PSF image
psfData=pyfits.getdata('data/psf_test.fits')
# setup the task
fpTask=fpfs.fpfsBase.fpfsTask(psfData)


# read GAL image
galImgAll=pyfits.getdata('data/gal_test.fits')
imgList=[galImgAll[i*64:(i+1)*64,0:64] for i in range(4)]

# measure FPFS moments
a=fpTask.measure(imgList)

# measure FPFS ellipticity, FPFS response

C=100 # the weighting parameter
b=fpfs.fpfsBase.fpfsM2E(a,C)
print(np.average(b['fpfs_e1'])/np.average(b['fpfs_RE']))
```

## Reference
