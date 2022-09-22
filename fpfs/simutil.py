# FPFS shear estimator
# Copyright 20210905 Xiangchong Li.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
# python lib

import os
import gc
import galsim
import logging
import numpy as np
import astropy.io.fits as pyfits

## For ring tests
def make_ringrot_radians(nord=8):
    """Generates rotation angle array for ring test

    Args:
        nord (int):             up to 1/2**nord*pi rotation
    Returns:
        rotArray (ndarray):     rotation array [in units of radians]
    """
    rotArray=   np.zeros(2**nord)
    nnum    =   0
    for j in range(nord+1):
        nj  =   2**j
        for i in range(1,nj,2):
            nnum+=1
            rotArray[nnum]=i/nj
    rotArray=rotArray*np.pi
    return rotArray

class sim_test():
    def __init__(self,shear,rng,scale=0.263,psf_fwhm=0.9,gal_hlr=0.5,ngrid=32):
        """Simulates an exponential object with moffat PSF, this class has the same
        observational setup as
        https://github.com/esheldon/ngmix/blob/38c379013840b5a650b4b11a96761725251772f5/examples/metacal/metacal.py#L199

        Args:
            shear (tuple):      tuple of [g1, g2]. The shear in each component
            rng (randState):    The random number generator
        """
        self.rng=   rng
        dx      =   0.5*scale
        dy      =   0.5*scale

        psf     =   galsim.Moffat(beta=2.5,fwhm=psf_fwhm,).shear(g1=0.02, g2=-0.02,)
        psf     =   psf.shift(
            dx  =   dx,
            dy  =   dy,
        )

        obj0    =   galsim.Exponential(
            half_light_radius=gal_hlr,
        ).shear(
            g1  =   shear[0],
            g2  =   shear[1],
        )

        self.scale= scale

        self.obj=   galsim.Convolve(psf, obj0)

        # define the psf and gal here which will be repeatedly used
        self.img0=  self.obj.drawImage(nx=ngrid,ny=ngrid,scale=scale).array
        self.psf=   psf.drawImage(nx=ngrid,ny=ngrid,scale=scale).array
        self.ngrid= ngrid
        return

    def make_image(self,noise,psf_noise=0.,do_shift=False):
        """Generates a galaxy image

        Args:
            noise (float):      Noise for the image
            psf_noise (float):  Noise for the PSF [defalut: 0.]
            do_shift (bool):    whether shift the galaxy [default: False]
        Returns:
            im (ndarray):       galaxy image
            psf_im (ndarray):   PSF image
        """
        if do_shift:
            dy, dx  =   self.rng.uniform(low=-self.scale/2, high=self.scale/2, size=2)
            obj     =   self.obj.shift(dx  = dx, dy  = dy)
            self.img=   obj.drawImage(nx=self.ngrid,ny=self.ngrid,scale=self.scale).array
        else:
            self.img=   self.img0
        if noise>1e-10:
            img =   self.img+self.rng.normal(scale=noise,size=self.img.shape)
        else:
            img =   self.img
        if psf_noise>1e-10:
            psf =   self.psf+self.rng.normal(scale=psf_noise, size=self.psf.shape)
        else:
            psf =   self.psf
        return img,psf

def coord_distort(x,y,xref,yref,gamma1,gamma2,kappa=0.):
    '''Distorts coordinates by shear

    Args:
        x (ndarray):    input coordinates [x]
        y (ndarray):    input coordinates [y]
        xref (float):   reference point [x]
        yref (float):   reference point [y]
        gamma1 (float): first component of shear distortion
        gamma2 (float): second component of shear distortion
        kappa (float):  kappa distortion [default: 0]
    Returns:
        x2 (ndarray):   distorted coordiantes [x]
        y2 (ndarray):   distorted coordiantes [y]
    '''
    xu  =   x-xref
    yu  =   y-yref
    x2  =   (1+kappa+gamma1)*xu+gamma2*yu+xref
    y2  =   gamma2*xu+(1+kappa-gamma1)*yu+yref
    return x2,y2

def coord_rotate(x,y,xref,yref,theta):
    '''Rotates coordinates by an angle theta

    Args:
        x (ndarray):    input coordinates [x]
        y (ndarray):    input coordinates [y]
        xref (float):   reference point [x]
        yref (float):   reference point [y]
        theta (float):  rotation angle [rads]
    Returns:
        x2 (ndarray):   rotated coordiantes [x]
        y2 (ndarray):   rotated coordiantes [y]
    '''
    xu  =   x-xref
    yu  =   y-yref
    x2  =   np.cos(theta)*xu-np.sin(theta)*yu+xref
    y2  =   np.sin(theta)*xu+np.cos(theta)*yu+yref
    return x2,y2

def make_cosmo_sim(outDir,incname,psfInt,gname,Id0,ny=5000,nx=5000,rfrac=0.46,scale=0.168,\
        do_write=True,return_array=False,magzero= 27.,rot2=0.):

    """Makes cosmo-like blended galaxy image simulations.

    Args:
        outDir (str):           output directory
        incname (str):          input catalog Name
        psfInt (PSF):           input PSF object of galsim
        gname (str):            shear distortion setup
        Id0 (int):              index of the simulation
        ny (int):               number of galaxies in y direction [default: 5000]
        nx (int):               number of galaxies in x direction [default: 5000]
        rfrac(float):           fraction of radius to minimum between nx and ny
        do_write (bool):        whether write output [default: True]
        return_array (bool):    whether return galaxy array [default: False]
        magzero (float):        magnitude zero point
        rot2 (float):           additional rotational angle [in units of radians]
    """
    np.random.seed(Id0)
    outFname=   os.path.join(outDir,'image-%d-%s.fits' %(Id0,gname))
    if os.path.isfile(outFname):
        logging.info('Already have the outcome.')
        if do_write:
            logging.info('Nothing to write.')
        if return_array:
            return pyfits.getdata(outFname)
        else:
            return None

    bigfft  =   galsim.GSParams(maximum_fft_size=10240) # galsim setup
    # Get the shear information
    # Three choice on g(-0.02,0,0.02)
    gList   =   np.array([-0.02,0.,0.02])
    gList   =   gList[[eval(i) for i in gname.split('-')[-1]]]

    # PSF
    scale=  0.168 #[arcsec]

    # number of galaxy
    # we only have `ngeff' galsim galaxies but with `nrot' rotation
    nrot    =   4
    r2      =   (min(nx,ny)*rfrac)**2.
    density =   int(outDir.split('_psf')[0].split('_cosmo')[-1])
    ngal    =   max(int(r2*np.pi*scale**2./3600.*density),nrot)
    ngal    =   int(ngal//nrot*nrot)
    ngeff   =   ngal//nrot
    logging.info('We have %d galaxies in total, and each %d are the same' %(ngal,nrot))

    # get the cosmos catalog
    inCat   =   pyfits.getdata(incname)
    ntrain  =   len(inCat)
    inds    =   np.random.randint(0,ntrain,ngeff)
    inCat   =   inCat[inds]

    # evenly distributed within a radius, min(nx,ny)*rfrac
    farray  =   2.*(np.random.randint(2,size=ngeff)-0.5) #-1 or 1
    rarray  =   np.sqrt(r2*np.random.rand(ngeff))*farray # radius
    tarray  =   np.random.uniform(0.,np.pi/nrot,ngeff)   # theta (0,pi/nrot)
    tarray  =   tarray+rot2
    xarray  =   rarray*np.cos(tarray)+nx//2     # x
    yarray  =   rarray*np.sin(tarray)+ny//2     # y
    rsarray =   np.random.uniform(0.95,1.05,ngeff)
    del rarray,tarray,farray

    zbound  =   np.array([1e-5,0.5477,0.8874,1.3119,12.0]) #sim 3

    gal_image=  galsim.ImageF(nx,ny,scale=scale)
    gal_image.setOrigin(0,0)
    for ii in range(ngal):
        ig      =   ii//nrot; irot    =   ii%nrot
        ss      =   inCat[ig]
        # x,y
        xi      =   xarray[ig]; yi    =   yarray[ig]
        # randomly rotate by an angle; we have 180/nrot deg pairs to remove
        # shape noise in additive bias estimation
        rotAng  =   np.pi/nrot*irot; ang     =   rotAng*galsim.radians
        xi,yi   =   coord_rotate(xi,yi,nx//2,ny//2,rotAng)
        # determine redshift
        gInd=   np.where((ss['zphot']>zbound[:-1])&(ss['zphot']<=zbound[1:]))[0]
        if len(gInd)==1:
            if gname.split('-')[0]=='g1':
                g1=gList[gInd][0]
                g2=0.
            elif gname.split('-')[0]=='g2':
                g1=0.
                g2=gList[gInd][0]
            else:
                raise ValueError('g1 or g2 must be in gname')
        else:
            g1  =   0.; g2  = 0.

        gal =   generate_cosmos_gal(ss,truncr=-1.,gsparams=bigfft)
        # determine and assign flux
        # HSC's i-band coadds zero point is 27
        flux=   10**((magzero-ss['mag_auto'])/2.5)
        gal =   gal.withFlux(flux)
        # rescale the radius while keeping the surface brightness the same
        gal =   gal.expand(rsarray[ig])
        # rotate by 'ang'
        gal =   gal.rotate(ang)
        # lensing shear
        gal =   gal.shear(g1=g1,g2=g2)
        # position and subpixel offset
        xi,yi=  coord_distort(xi,yi,nx//2,ny//2,g1,g2)
        xu  =   int(xi);yu  =   int(yi)
        dx  =   (0.5+xi-xu)*scale;dy  =   (0.5+yi-yu)*scale
        gal =   gal.shift(dx,dy)
        # PSF
        gal =   galsim.Convolve([psfInt,gal],gsparams=bigfft)
        # Bounary
        gPix=   max(gal.getGoodImageSize(scale),32)
        rx1 =   np.min([gPix,xu])
        rx2 =   np.min([gPix,nx-xu-1])
        rx  =   int(min(rx1,rx2))
        del rx1,rx2
        ry1 =   np.min([gPix,yu])
        ry2 =   np.min([gPix,ny-yu-1])
        ry  =   int(min(ry1,ry2))
        del ry1,ry2
        # draw galaxy
        b   =   galsim.BoundsI(xu-rx,xu+rx-1,yu-ry,yu+ry-1)
        sub_img =   gal_image[b]
        gal.drawImage(sub_img,add_to_image=True)
        # print(galsim.hsm.FindAdaptiveMom(gal_image))
        del gal,b,sub_img,xu,yu,xi,yi,gPix
    gc.collect()
    del inCat,psfInt
    if do_write:
        gal_image.write(outFname,clobber=True)
    if return_array:
        return gal_image.array

def generate_cosmos_gal(record,truncr=5.,gsparams=None):
    """Generates COSMOS galaxies; modified version of
    https://github.com/GalSim-developers/GalSim/blob/releases/2.3/galsim/scene.py#L626

    Args:
        record (ndarray):   one row of the COSMOS galaxy catalog
        truncr (float):     truncation ratio
        gsparams:           An GSParams argument.
    Returns:
        gal:    Galsim galaxy
    """
    # record columns:
    # For 'sersicfit', the result is an array of 8 numbers for each:
    #     SERSICFIT[0]: intensity of light profile at the half-light radius.
    #     SERSICFIT[1]: half-light radius measured along the major axis, in
    #                   units of pixels in the COSMOS lensing data reductions
    #                   (0.03 arcsec).
    #     SERSICFIT[2]: Sersic n.
    #     SERSICFIT[3]: q, the ratio of minor axis to major axis length.
    #     SERSICFIT[4]: boxiness, currently fixed to 0, meaning isophotes are all
    #                   elliptical.
    #     SERSICFIT[5]: x0, the central x position in pixels.
    #     SERSICFIT[6]: y0, the central y position in pixels.
    #     SERSICFIT[7]: phi, the position angle in radians. If phi=0, the major
    #                   axis is lined up with the x axis of the image.
    # For 'bulgefit', the result is an array of 16 parameters that comes from doing a
    # 2-component sersic fit.  The first 8 are the parameters for the disk, with n=1, and
    # the last 8 are for the bulge, with n=4.

    bparams = record['bulgefit']
    sparams = record['sersicfit']
    use_bulgefit = record['use_bulgefit']
    if use_bulgefit:
        # Bulge parameters:
        # Minor-to-major axis ratio:
        bulge_q = bparams[11]
        # Position angle, now represented as a galsim.Angle:
        bulge_beta = bparams[15]*galsim.radians
        disk_q  = bparams[3]
        disk_beta = bparams[7]*galsim.radians
        bulge_hlr = record['hlr'][1]
        bulge_flux = record['flux'][1]
        disk_hlr= record['hlr'][2]
        disk_flux = record['flux'][2]
        if truncr<=0.99:
            btrunc=None
            # Then combine the two components of the galaxy.
            bulge = galsim.DeVaucouleurs(flux=bulge_flux, \
                    half_light_radius=bulge_hlr,\
                    gsparams=gsparams)
            disk = galsim.Exponential(flux=disk_flux, \
                    half_light_radius=disk_hlr,
                    gsparams=gsparams)
        else:
            btrunc=bulge_hlr*truncr
            # Then combine the two components of the galaxy.
            bulge = galsim.DeVaucouleurs(flux=bulge_flux, \
                    half_light_radius=bulge_hlr,trunc=btrunc,\
                    gsparams=gsparams)
            dtrunc=disk_hlr*truncr
            disk = galsim.Sersic(1., flux=disk_flux, \
                    half_light_radius=disk_hlr,trunc=dtrunc,
                    gsparams=gsparams)
        # Apply shears for intrinsic shape.
        if bulge_q < 1.:  # pragma: no branch
            bulge = bulge.shear(q=bulge_q, beta=bulge_beta)
        if disk_q < 1.:  # pragma: no branch
            disk = disk.shear(q=disk_q, beta=disk_beta)
        gal = bulge + disk
    else:
        # Do a similar manipulation to the stored quantities for the single Sersic profiles.
        gal_n = sparams[2]
        # Fudge this if it is at the edge of the allowed n values.  Since GalSim (as of #325 and
        # #449) allow Sersic n in the range 0.3<=n<=6, the only problem is that the fits
        # occasionally go as low as n=0.2.  The fits in this file only go to n=6, so there is no
        # issue with too-high values, but we also put a guard on that side in case other samples
        # are swapped in that go to higher value of sersic n.
        if gal_n < 0.3: gal_n = 0.3
        if gal_n > 6.0: gal_n = 6.0

        # GalSim is much more efficient if only a finite number of Sersic n values are used.
        # This (optionally given constructor args) rounds n to the nearest 0.05.
        # change to 0.1 to speed up
        gal_n = galsim_round_sersic(gal_n, 0.1)
        gal_hlr = record['hlr'][0]
        gal_flux = record['flux'][0]
        if truncr<=0.99:
            btrunc=None
            gal = galsim.Sersic(gal_n, flux=gal_flux, \
                    half_light_radius=gal_hlr,
                    gsparams=gsparams)
        else:
            btrunc=gal_hlr*truncr
            gal = galsim.Sersic(gal_n, flux=gal_flux, \
                    half_light_radius=gal_hlr,trunc=btrunc,
                    gsparams=gsparams)
    return gal

def galsim_round_sersic(n, sersic_prec):
    return float(int(n/sersic_prec + 0.5)) * sersic_prec

def make_basic_sim(outDir,incname,psfInt,gname,Id0,ny=6400,nx=6400,scale=0.168,\
        do_write=True,return_array=False,magzero=27.,rot2=0):
    """Makes basic **isolated** galaxy image simulation.

    Args:
        outDir (str):           output directory
        incname (str):          input catalog Name
        psfInt (PSF):           input PSF object of galsim
        gname (str):            shear distortion setup
        Id0 (int):              index of the simulation
        ny (int):               number of pixels in y direction
        nx (int):               number of pixels in y direction
        do_write (bool):        whether write output [default: True]
        return_array (bool):    whether return galaxy array [default: False]
        magzero (float):        magnitude zero point [27 for HSC]
        rot2 (float):           additional rotation angle
    """
    outFname=   os.path.join(outDir,'image-%d-%s.fits' %(Id0,gname))
    if os.path.isfile(outFname):
        logging.info('Already have the outcome.')
        if do_write:
            logging.info('Nothing to write.')
        if return_array:
            return pyfits.getdata(outFname)
        else:
            return None

    # Basic parameters
    ngrid   =   64
    ngalx   =   nx//64
    ngaly   =   ny//64
    ngal    =   ngalx*ngaly
    # Get the shear information
    gList   =   np.array([-0.02,0.,0.02])
    gList   =   gList[[eval(i) for i in gname.split('-')[-1]]]
    if gname.split('-')[0]=='g1':
        g1  =    gList[0]
        g2  =    0.
    elif gname.split('-')[0]=='g2':
        g1  =    0.
        g2  =    gList[0]
    else:
        raise ValueError('cannot decide g1 or g2')
    logging.info('Processing for %s, and shears for four redshift bins are %s.' %(gname,gList))

    gal_image=  galsim.ImageF(nx,ny,scale=scale)
    gal_image.setOrigin(0,0)
    bigfft  =   galsim.GSParams(maximum_fft_size=10240)
    if 'basic' in outDir:
        if Id0>= 4000:
            logging.info('galaxy image index greater than 8000' )
            return
        np.random.seed(Id0)
        logging.info('Making Basic Simulation. ID: %d' %(Id0))
        # Galsim galaxies
        # directory   =   os.path.join(os.environ['homeWrk'],\
        #                 'COSMOS/galsim_train/COSMOS_25.2_training_sample/')
        # assert os.path.isdir(directory), 'cannot find galsim galaxies'
        # catName     =   'real_galaxy_catalog_25.2.fits'
        # cosmos_cat  =   galsim.COSMOSCatalog(catName,dir=directory)
        # catalog
        inCat   =   pyfits.getdata(incname)
        ntrain  =   len(inCat)
        nrot    =   2
        ngeff   =   ngal//nrot
        inds    =   np.random.randint(0,ntrain,ngeff)
        inCat   =   inCat[inds]
        gal0    =   None
        for i in range(ngal):
            # boundary
            ix  =   i%ngalx
            iy  =   i//ngalx
            b   =   galsim.BoundsI(ix*ngrid,(ix+1)*ngrid-1,iy*ngrid,(iy+1)*ngrid-1)
            # each galaxy
            ig  =   i//nrot;   irot =i%nrot
            ss  =   inCat[ig]
            if irot==0:
                del gal0
                # gal0  =  cosmos_cat.makeGalaxy(gal_type='parametric',\
                #             index=ss['index'],gsparams=bigfft)
                gal0    =   generate_cosmos_gal(ss,truncr=5.,gsparams=bigfft)
                # accounting for zeropoint difference between COSMOS HST and HSC
                # HSC's i-band coadds zero point is 27
                flux    =   10**((magzero-ss['mag_auto'])/2.5)
                # flux_scaling=   2.587
                gal0    =   gal0.withFlux(flux)
                # rescale the radius by 'rescale' and keep surface brightness the same
                rescale =   np.random.uniform(0.95,1.05)
                gal0    =   gal0.expand(rescale)
                # rotate by 'ang'
                ang     =   (np.random.uniform(0.,np.pi*2.)+rot2)*galsim.radians
                gal0    =   gal0.rotate(ang)
            else:
                assert gal0 is not None
                ang     =   np.pi/nrot*galsim.radians
                # update gal0
                gal0    =   gal0.rotate(ang)
            # shear distortion
            gal =   gal0.shear(g1=g1,g2=g2)
            # shift galaxy
            if 'Shift' in outDir:
                dx  =   np.random.uniform(-0.5,0.5)*scale
                dy  =   np.random.uniform(-0.5,0.5)*scale
                gal =   gal.shift(dx,dy)
            # shift to (ngrid//2,ngrid//2)
            gal =   gal.shift(0.5*scale,0.5*scale)
            gal =   galsim.Convolve([psfInt,gal],gsparams=bigfft)
            # draw galaxy
            sub_img =   gal_image[b]
            gal.drawImage(sub_img,add_to_image=True)
            del gal,b,sub_img,ss
        del inCat
        gc.collect()
    elif 'small' in outDir:
        ud      =   galsim.UniformDeviate(Id0)
        # use galaxies with random knots
        # we only support three versions of small galaxies with different radius
        irr =   int(outDir.split('_psf')[0].split('small')[-1])
        if irr==0:
            radius  =0.07
        elif irr==1:
            radius  =0.15
        elif irr==2:
            radius  =0.20
        else:
            raise ValueError('Something wrong with the outDir! we only support'
                    'three versions of small galaxies')
        logging.info('Making Small Simulation with Random Knots.' )
        logging.info('Radius: %s, ID: %s.' %(radius,Id0) )
        npoints =   20
        gal0    =   galsim.RandomKnots(half_light_radius=radius,\
                    npoints=npoints,flux=10.,rng=ud)
        for ix in range(100):
            for iy in range(100):
                igal   =    ix*100+iy
                b      =    galsim.BoundsI(ix*ngrid,(ix+1)*ngrid-1,\
                            iy*ngrid,(iy+1)*ngrid-1)
                if igal%4==0 and igal!=0:
                    gal0=   galsim.RandomKnots(half_light_radius=radius,\
                            npoints=npoints,flux=10.,rng=ud,gsparams=bigfft)
                sub_img =   gal_image[b]
                ang     =   igal%4*np.pi/4. * galsim.radians
                gal     =   gal0.rotate(ang)
                # Shear the galaxy
                gal     =   gal.shear(g1=g1,g2=g2)
                gal     =   galsim.Convolve([psfInt,gal],gsparams=bigfft)
                # Draw the galaxy image
                gal.drawImage(sub_img,add_to_image=True)
                del gal,b,sub_img
                gc.collect()
        del ud
        gc.collect()
    else:
        raise ValueError("outDir should cotain 'basic' or 'small'!!")
    if do_write:
        gal_image.write(outFname,clobber=True)
    if return_array:
        return gal_image.array

def make_noise_sim(outDir,infname,Id0,ny=6400,nx=6400,scale=0.168,\
        do_write=True,return_array=False):
    """Makes pure noise for galaxy image simulation.

    Args:
        outDir (str):           output directory
        incname (str):          input correlation function name
        Id0 (int):              index of the simulation
        ny (int):               number of pixels in y direction
        nx (int):               number of pixels in x direction
        do_write (bool):        whether write output [default: True]
        return_array (bool):    whether return galaxy array [default: False]
    """
    logging.info('begining for field %04d' %(Id0))
    outFname    =   os.path.join(outDir,'noi%04d.fits' %(Id0))
    if os.path.exists(outFname):
        if do_write:
            logging.info('Nothing to write.')
        if return_array:
            return pyfits.getdata(outFname)
        else:
            return None
    logging.info('simulating noise for field %s' %(Id0))
    variance    =   0.01
    ud          =   galsim.UniformDeviate(Id0*10000+1)

    # setup the galaxy image and the noise image
    noi_image   =   galsim.ImageF(nx,ny,scale=scale)
    noi_image.setOrigin(0,0)
    corNoise    =   galsim.getCOSMOSNoise(file_name=infname,\
                    rng=ud,cosmos_scale=scale,variance=variance)
    corNoise.applyTo(noi_image)
    if do_write:
        pyfits.writeto(outFname,noi_image.array)
    if return_array:
        return noi_image.array
    return

def make_gal_ssbg(shear,psf,rng,r1,r0=20.):
    """This function is for the simulation for source photon noise. It
    simulates an exponential object with moffat PSF, given a SNR [r0] and a
    source background noise ratio [r0].

    Args:
        shear (tuple):          [g1, g2]. The shear in each component
        rng (randState):        The random number generator
        r1  (float):            The source background noise variance ratio
        r0  (float):            The SNR of galaxy
        psf (galsim.Moffat):    a Moffat PSF

    Returns:
       img (ndarray):           noisy image array
    """
    scale   =   0.263
    gal_hlr =   0.5

    dy, dx  =   rng.uniform(low=-scale/2, high=scale/2, size=2)

    obj0    =   galsim.Exponential(
        half_light_radius=gal_hlr,
    ).shear(
        g1  =   shear[0],
        g2  =   shear[1],
    ).shift(
        dx  =   dx,
        dy  =   dy,
    )
    obj     =   galsim.Convolve(psf, obj0)

    # define the psf and psf here which will be repeatedly used
    psf     =   psf.drawImage(scale=scale).array
    # galaxy image:
    img     =   obj.drawImage(scale=scale).array
    ngrid   =   img.shape[0]
    # noise image:
    noimg   =   rng.normal(scale=1.,size=img.shape)
    # get the current flux using the 5x5 substamps centered at the stamp's center
    flux_tmp=   np.sum(img[ngrid//2-2:ngrid//2+3,ngrid//2-2:ngrid//2+3])
    # the current (expectation of) total noise std on the 5x5 substamps is 5 since for each
    # pixel, the expecatation value of variance is 1; therefore, the expectation value of variance is 25...
    std_tmp =   5
    # normalize both the galaxy image and noise image so that they will have
    # flux=1 and variance=1 (expectation value) in the 5x5 substamps
    img     =   img/flux_tmp
    noimg   =   noimg/std_tmp
    # now we can determine the flux and background variance using equation (3)
    F       =   r0**2.*(1+r1)/r1
    B       =   F/r1
    img     =   img*F
    noimg   =   noimg*np.sqrt(B)
    img     =   img+noimg
    return img

