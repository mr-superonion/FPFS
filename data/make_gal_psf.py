import galsim
import numpy as np
import astropy.io.fits as pyfits
r'''
This code simulate four noiseless galaxies with 45 degree difference in
orientation and distort by gamma=(0.02,0) to test the FPFS code.
'''
ngrid       =   64
nx          =   1
ny          =   4
ndata       =   nx*ny
nrot        =   4
scale       =   0.168
npoints     =   100
ud          =   galsim.UniformDeviate(10000)
bigfft      =   galsim.GSParams(maximum_fft_size=10240)


psfInt=galsim.Moffat(beta=3.5,fwhm=0.615,trunc=0.615*4.)
psfInt=psfInt.shear(e1=0.,e2=0.02)
psfImg   =   psfInt.drawImage(nx=64,ny=64,scale=scale)

galImg  =   galsim.ImageF(nx*ngrid,ny*ngrid,scale=scale)
galImg.setOrigin(0,0)

i           =   0
while i <ndata:
    # Prepare the subimage
    ix      =   i%nx
    iy      =   i//nx
    b       =   galsim.BoundsI(ix*ngrid,(ix+1)*ngrid-1,iy*ngrid,(iy+1)*ngrid-1)
    sub_image=  galImg[b]
    #simulate the galaxy
    if i%nrot==0:
        # update galaxy
        gal0    =   galsim.RandomKnots(half_light_radius=0.2,\
                npoints=npoints,flux=1.)
        gal0    =   gal0.shear(e1=0.2,e2=-0.05)

        # rotate the galaxy
        ang     =   ud()*2.*np.pi * galsim.radians
        gal0    =   gal0.rotate(ang)
    else:
        gal0    =   gal0.rotate(1./nrot*np.pi*galsim.radians)
    gal1        =   gal0.shear(g1=0.02,g2=0.)
    final1      =   galsim.Convolve([psfInt,gal1],gsparams=bigfft)
    final1.drawImage(sub_image)
    i   +=  1

galFname    =   'gal_test.fits'
pyfits.writeto(galFname,galImg.array,overwrite=True)
psfFname    =   'psf_test.fits'
pyfits.writeto(psfFname,psfImg.array,overwrite=True)
