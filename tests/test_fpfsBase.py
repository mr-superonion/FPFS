import fpfs
import numpy as np
import astropy.io.fits as pyfits

def test_noiseless_gals():
    # Read PSF image
    psfData=pyfits.getdata('../data/psf_test.fits')
    # Setup the FPFS task.
    # For noiseless galaxies, no need to input
    # model for noise power.
    fpTask=fpfs.fpfsBase.fpfsTask(psfData)


    # Read GAL image
    galImgAll=pyfits.getdata('../data/gal_test.fits')
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
    np.testing.assert_almost_equal(g_est, 0.02, 5)
    return

if __name__ == '__main__':
    test_noiseless_gals()

