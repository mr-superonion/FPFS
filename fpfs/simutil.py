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
import numpy.lib.recfunctions as rfn
try:
    import fitsio
    with_hst=True
    hpInfofname     =   os.path.join(os.environ['homeWrk'],'skyMap/healpix-nside%d-nest.fits')
    cosmoHSThpix    =   np.array(\
           [1743739, 1743740, 1743741, 1743742, 1743743, 1743825, 1743828,
            1743829, 1743830, 1743831, 1743836, 1743837, 1744397, 1744398,
            1744399, 1744402, 1744408, 1744409, 1744410, 1744411, 1744414,
            1744416, 1744417, 1744418, 1744419, 1744420, 1744421, 1744422,
            1744423, 1744424, 1744425, 1744426, 1744427, 1744428, 1744429,
            1744430, 1744431, 1744432, 1744433, 1744434, 1744435, 1744436,
            1744437, 1744438, 1744439, 1744440, 1744441, 1744442, 1744443,
            1744444, 1744445, 1744446, 1744447, 1744482, 1744488, 1744489,
            1744490, 1744491, 1744494, 1744512, 1744513, 1744514, 1744515,
            1744516, 1744517, 1744518, 1744519, 1744520, 1744521, 1744522,
            1744523, 1744524, 1744525, 1744526, 1744527, 1744528, 1744529,
            1744530, 1744531, 1744532, 1744533, 1744534, 1744535, 1744536,
            1744537, 1744538, 1744539, 1744540, 1744541, 1744542, 1744543,
            1744545, 1744548, 1744549, 1744550, 1744551, 1744557, 1744560,
            1744561, 1744562, 1744563, 1744564, 1744565, 1744566, 1744567,
            1744568, 1744569, 1744570, 1744571, 1744572, 1744573, 1744574,
            1744576, 1744577, 1744578, 1744579, 1744580, 1744581, 1744582,
            1744583, 1744584, 1744585, 1744586, 1744587, 1744588, 1744589,
            1744590, 1744594, 1744608, 1744609, 1744610, 1750033])
except (ImportError, KeyError) as error:
    with_hst=False

if with_hst:
    class cosmoHSTGal():
        def __init__(self,version):
            self.hpInfo     =   fitsio.read(hpInfofname %512)
            self.directory  =   os.path.join(os.environ['homeWrk'],'COSMOS/galsim_train/COSMOS_25.2_training_sample/')
            self.catName    =   'real_galaxy_catalog_25.2.fits'
            self.finName    =   os.path.join(self.directory,'cat_used.fits')
            if version=='252':
                self.hpDir  =   os.path.join(self.directory,'healpix-nside512')
            elif version=='252E':
                _dir        =   os.path.join(os.environ['homeWrk'],\
                            'COSMOS/galsim_train/COSMOS_25.2_extended/')
                self.hpDir  =   os.path.join(_dir,'healpix-nside512')
            else:
                return
            return

        def selectHpix(self,pixId):
            """
            # select galaxies in one healPIX
            """
            indFname    =   os.path.join(self.hpDir,'%d-25.2_ind.fits' %pixId)
            if os.path.isfile(indFname):
                __mask  =   fitsio.read(indFname)
            else:
                dd      =   self.hpInfo[self.hpInfo['pix']==pixId]
                __mask  =   (self.catused['ra']>dd['raMin'])\
                        &(self.catused['ra']<dd['raMax'])\
                        &(self.catused['dec']>dd['decMin'])\
                        &(self.catused['dec']<dd['decMax'])
            __out   =   self.catused[__mask]
            return __out

        def readHpixSample(self,pixId):
            """
            # select galaxies in one healPIX
            """
            fname   =   os.path.join(self.hpDir,'cat-%d-25.2.fits' %pixId)
            if os.path.isfile(fname):
                out =   fitsio.read(fname)
            else:
                out =   None
            return out

        def readHSTsample(self):
            """
            # read the HST galaxy training sample
            """
            if os.path.isfile(self.finName):
                catfinal    =   fitsio.read(self.finName)
            else:
                cosmos_cat  =   galsim.COSMOSCatalog(self.catName,dir=self.directory)
                # used index
                index_use   =   cosmos_cat.orig_index
                # used catalog
                paracat     =   cosmos_cat.param_cat[index_use]
                # parametric catalog
                oricat      =   fitsio.read(cosmos_cat.real_cat.getFileName())[index_use]
                ra          =   oricat['RA']
                dec         =   oricat['DEC']
                indexNew    =   np.arange(len(ra),dtype=int)
                __tmp=np.stack([ra,dec,indexNew]).T
                radec=np.array([tuple(__t) for __t in __tmp],dtype=[('ra','>f8'),('dec','>f8'),('index','i8')])
                catfinal    =   rfn.merge_arrays([paracat,radec], flatten = True, usemask = False)
                fitsio.write(self.finName,catfinal)
            self.catused    =   catfinal
            return

# LSST Task
try:
    import lsst.geom as geom
    import lsst.afw.math as afwMath
    import lsst.afw.image as afwImg
    import lsst.afw.geom as afwGeom
    import lsst.meas.algorithms as meaAlg
    with_lsst=True
except ImportError as error:
    with_lsst=False

if with_lsst:
    def makeLsstExposure(galData,psfData,pixScale,variance):
        """
        make an LSST exposure object

        Args:
            galData (ndarray):  array of galaxy image
            psfData (ndarray):  array of PSF image
            pixScale (float):   pixel scale
            variance (float):   noise variance

        Returns:
            exposure:   LSST exposure object
        """
        if not with_lsst:
            raise ImportError('Do not have lsstpipe!')
        ny,nx       =   galData.shape
        exposure    =   afwImg.ExposureF(nx,ny)
        exposure.getMaskedImage().getImage().getArray()[:,:]=galData
        exposure.getMaskedImage().getVariance().getArray()[:,:]=variance
        #Set the PSF
        ngridPsf    =   psfData.shape[0]
        psfLsst     =   afwImg.ImageF(ngridPsf,ngridPsf)
        psfLsst.getArray()[:,:]= psfData
        psfLsst     =   psfLsst.convertD()
        kernel      =   afwMath.FixedKernel(psfLsst)
        kernelPSF   =   meaAlg.KernelPsf(kernel)
        exposure.setPsf(kernelPSF)
        #prepare the wcs
        #Rotation
        cdelt   =   (pixScale*geom.arcseconds)
        CD      =   afwGeom.makeCdMatrix(cdelt, geom.Angle(0.))#no rotation
        #wcs
        crval   =   geom.SpherePoint(geom.Angle(0.,geom.degrees),geom.Angle(0.,geom.degrees))
        #crval   =   afwCoord.IcrsCoord(0.*afwGeom.degrees, 0.*afwGeom.degrees) # hscpipe6
        crpix   =   geom.Point2D(0.0, 0.0)
        dataWcs =   afwGeom.makeSkyWcs(crpix,crval,CD)
        exposure.setWcs(dataWcs)
        #prepare the frc
        dataCalib = afwImg.makePhotoCalibFromCalibZeroPoint(63095734448.0194)
        exposure.setPhotoCalib(dataCalib)
        return exposure

## For ring tests
def make_ringrot_radians(nord=8):
    """
    Generate rotation angle array for ring test

    Args:
        nord (int):
            up to 1/2**nord*pi rotation

    Returns:
        rotArray (ndarray):
            rotation array [in units of radians]
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
        """
        simulate an exponential object with moffat PSF, this class has the same observational setup as
        https://github.com/esheldon/ngmix/blob/38c379013840b5a650b4b11a96761725251772f5/examples/metacal/metacal.py#L199

        Args:
            shear (tuple):      (g1, g2),The shear in each component
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
        """
        generate a galaxy image

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
    '''
    Args:
        x (ndarray):    input coordinates (x)
        y (ndarray):    input coordinates (y)
        xref (float):   reference point (x)
        yref (float):   reference point (y)
        gamma1 (float): first component of shear distortion
        gamma2 (float): second component of shear distortion
        kappa (float):  kappa distortion (default: 0)
    Returns:
        x2 (ndarray):   distorted coordiantes (x)
        y2 (ndarray):   distorted coordiantes
    '''
    xu  =   x-xref
    yu  =   y-yref
    x2  =   (1+kappa+gamma1)*xu+gamma2*yu+xref
    y2  =   gamma2*xu+(1+kappa-gamma1)*yu+yref
    return x2,y2

def make_cosmo_sim(outDir,gname,Id0,ny=5000,nx=5000,do_write=True,return_array=False):
    """Makes cosmo galaxy image simulation (blended)
    Args:
        outDir (str):
            output directory
        gname (str):
            shear distortion setup
        Id0 (int):
            index of the simulation
        ny (int):
            number of galaxies in y direction (default: 5000)
        nx (int):
            number of galaxies in x direction (default: 5000)
        do_write (bool):
            whether write output (default: True)
        return_array (bool):
            whether return galaxy array (default: False)
    """
    if Id0>= 1024:
        logging.info('galaxy image index greater than 1024' )
        return
    eta     =   7
    irot    =   (Id0//8)%(2**eta)
    cgid    =   int(Id0%8)
    np.random.seed(cgid)
    outFname=   os.path.join(outDir,'image-%d-%s.fits' %(Id0,gname))
    if os.path.isfile(outFname):
        logging.info('Already have the outcome.')
        if do_write:
            logging.info('Nothing to write.')
        if return_array:
            return fitsio.read(outFname)
        else:
            return None

    rotArray    =   make_ringrot_radians(eta)
    # 2**7*8=1024 galaxy ID
    # 2**7 different rotations and dilations
    # for each galaxy ID 10000 parametric galaxies
    logging.info('We have %d rotation realizations' %len(rotArray))
    ang     =   rotArray[irot]*galsim.radians
    ud      =   galsim.UniformDeviate(Id0*212)
    rescale =   1.+(ud()-0.5)*0.1
    # Galsim galaxies
    directory   =   os.path.join(os.environ['homeWrk'],'COSMOS/galsim_train/COSMOS_25.2_training_sample/')
    catName     =   'real_galaxy_catalog_25.2.fits'
    cosmos_cat  =   galsim.COSMOSCatalog(catName,dir=directory)

    # Basic parameters
    bigfft      =   galsim.GSParams(maximum_fft_size=10240)
    # Get the shear information
    # Three choice on g(-0.02,0,0.02)
    gList   =   np.array([-0.02,0.,0.02])
    gList   =   gList[[eval(i) for i in gname.split('-')[-1]]]

    # PSF
    pix_scale=  0.168 #[arcsec]
    if 'HSC' in outDir:
        psfFname=   os.path.join(outDir,'psf-HSC.fits')
        assert os.path.isfile(psfFname), 'Cannot find input HSC PSF files'
        psfImg  =   galsim.fits.read(psfFname)
        psfInt  =   galsim.InterpolatedImage(psfImg,scale=pix_scale,flux = 1.)
        logging.info('Using HSC PSF')
    else:
        psfFWHM =   int(outDir.split('_psf')[-1])/100.
        psfInt  =   galsim.Moffat(beta=3.5,fwhm=psfFWHM,trunc=psfFWHM*4.)
        psfInt  =   psfInt.shear(e1=0.02,e2=-0.02)
        logging.info('Using Moffat PSF with FWHM: %s arcsec'%psfFWHM)

    # catalog
    density =   int(outDir.split('_psf')[0].split('_cosmo')[-1])
    ngal    =   max(int(nx*ny*0.93**2.*pix_scale**2./3600.*density),1)
    logging.info('We have %d galaxies in total' %ngal)
    cosmo252=   cosmoHSTGal('252')
    cosmo252.readHSTsample()
    ntrain  =   len(cosmo252.catused)
    inds    =   np.random.randint(0,ntrain,ngal)
    inCat   =   cosmo252.catused[inds]

    xarray  =   nx*0.92*np.random.rand(ngal)+nx*0.04
    yarray  =   ny*0.92*np.random.rand(ngal)+ny*0.04
    # if True:
    #     xarray  =   np.array([36.3])
    #     yarray  =   np.array([30.8])


    #zbound  =   np.array([0.,0.561,0.906,1.374,5.410]) #before sim3
    #zbound  =   np.array([0.005,0.5477,0.8874,1.3119,3.0]) #sim 3
    zbound  =   np.array([0.0001,0.5477,0.8874,1.3119,12.0]) #sim 3
    gal_image   =   galsim.ImageF(nx,ny,scale=pix_scale)
    gal_image.setOrigin(0,0)

    for ii in range(ngal):
        ss  =   inCat[ii]
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
        # each galaxy
        gal =   cosmos_cat.makeGalaxy(gal_type='parametric',index=ss['index'],gsparams=bigfft)
        flux=   10**((27.-ss['mag_auto'])/2.5)
        gal =   gal.withFlux(flux)

        # rescale the radius while keeping the surface brightness the same
        gal =   gal.expand(rescale)
        # rotate by 'ang'
        gal =   gal.rotate(ang)
        # lensing shear
        gal =   gal.shear(g1=g1,g2=g2)

        xi  =   xarray[ii]
        yi  =   yarray[ii]
        xi,yi=  coord_distort(xi,yi,nx//2,ny//2,g1,g2)
        xu  =   int(xi)
        yu  =   int(yi)
        dx  =   (0.5+xi-xu)*pix_scale
        dy  =   (0.5+yi-yu)*pix_scale
        gal =   gal.shift(dx,dy)
        # PSF
        gal =   galsim.Convolve([psfInt,gal],gsparams=bigfft)
        gPix=   gal.getGoodImageSize(pix_scale)
        rx1 =   np.min([gPix//1.0,xi])
        rx2 =   np.min([gPix//1.0,nx-xu-1])
        rx  =   int(min(rx1,rx2))
        del rx1,rx2
        ry1 =   np.min([gPix//1.0,yi])
        ry2 =   np.min([gPix//1.0,ny-yu-1])
        ry  =   int(min(ry1,ry2))
        del ry1,ry2
        # draw galaxy
        b   =   galsim.BoundsI(xu-rx,xu+rx-1,yu-ry,yu+ry-1)
        try:
            sub_img =   gal_image[b]
        except:
            print(yu,ry,xi)
            print(xu-rx,xu+rx-1,yu-ry,yu+ry-1)
            return
        del xu,yu
        gal.drawImage(sub_img,add_to_image=True)
        del gal,b,sub_img
        gc.collect()
    # print(galsim.hsm.FindAdaptiveMom(gal_image))
    del inCat,cosmos_cat,psfInt
    if do_write:
        gal_image.write(outFname,clobber=True)
    if return_array:
        return gal_image.array

def make_basic_sim(outDir,gname,Id0,ny=100,nx=100,do_write=True,return_array=False):
    """Makes basic galaxy image simulation (isolated)
    Args:
        outDir (str):
            output directory
        gname (str):
            shear distortion setup
        Id0 (int):
            index of the simulation
        ny (int):
            number of galaxies in y direction
        nx (int):
            number of galaxies in x direction
        do_write (bool):
            whether write output [default: True]
        return_array (bool):
            whether return galaxy array [default: False]
    """
    outFname=   os.path.join(outDir,'image-%d-%s.fits' %(Id0,gname))
    if os.path.isfile(outFname):
        logging.info('Already have the outcome.')
        if do_write:
            logging.info('Nothing to write.')
        if return_array:
            return fitsio.read(outFname)
        else:
            return None

    ngrid  =   64
    scale  =   0.168
    # Get the shear information
    gList  =   np.array([-0.02,0.,0.02])
    gList  =   gList[[eval(i) for i in gname.split('-')[-1]]]
    if gname.split('-')[0]=='g1':
        g1 =    gList[0]
        g2 =    0.
    elif gname.split('-')[0]=='g2':
        g1 =    0.
        g2 =    gList[0]
    else:
        raise ValueError('cannot decide g1 or g2')
    logging.info('Processing for %s, and shears for four redshift bins are %s.' %(gname,gList))
    # PSF
    psfFWHM =   int(outDir.split('_psf')[-1])/100.
    logging.info('The FWHM for PSF is: %s arcsec'%psfFWHM)
    psfInt  =   galsim.Moffat(beta=3.5,fwhm=psfFWHM,trunc=psfFWHM*4.)
    psfInt  =   psfInt.shear(e1=0.02,e2=-0.02)
    #psfImg =   psfInt.drawImage(nx=45,ny=45,scale=scale)

    gal_image=  galsim.ImageF(nx*ngrid,ny*ngrid,scale=scale)
    gal_image.setOrigin(0,0)
    ud      =   galsim.UniformDeviate(Id0+212)
    bigfft  =   galsim.GSParams(maximum_fft_size=10240)
    if 'basic' in outDir:
        if Id0>= 2048:
            logging.info('galaxy image index greater than 2048' )
            return
        rotArray    =   make_ringrot_radians(7)
        # 2**7*8=1024 galaxy ID
        # 2**7 different rotations and dilations
        # for each galaxy ID 10000 parametric galaxies
        logging.info('We have %d rotation realizations' %len(rotArray))
        irot    =   (Id0//8)%128
        ang     =   rotArray[irot]*galsim.radians
        rescale =   1.+(ud()-0.5)*0.1
        logging.info('%s' %rescale)
        # cosmos group ID =0...7
        # we use 80000 galsim galaxies repeatedly
        cgid    =   int(Id0%8)
        logging.info('Making Basic Simulation. ID: %d, cosmos GID: %d.' %(Id0,cgid))
        logging.info('The rotating angle is %.2f radians.' %rotArray[irot])
        # Galsim galaxies
        directory   =   os.path.join(os.environ['homeWrk'],\
                        'COSMOS/galsim_train/COSMOS_25.2_training_sample/')
        assert os.path.isdir(directory), 'cannot find galsim galaxies'
        catName     =   'real_galaxy_catalog_25.2.fits'
        cosmos_cat  =   galsim.COSMOSCatalog(catName,dir=directory)

        # Basic parameters
        flux_scaling=   2.587
        # catalog
        cosmo252=   cosmoHSTGal('252')
        cosmo252.readHSTsample()
        # cgid=0...7
        inCat  =   cosmo252.catused[cgid*nx*ny:(cgid+1)*nx*ny]
        for i,ss  in enumerate(inCat):
            ix  =   i%nx
            iy  =   i//nx
            b   =   galsim.BoundsI(ix*ngrid,(ix+1)*ngrid-1,iy*ngrid,(iy+1)*ngrid-1)
            # each galaxy
            gal =   cosmos_cat.makeGalaxy(gal_type='parametric',\
                    index=ss['index'],gsparams=bigfft)
            # rescale the radius by 'rescale' and keep surface brightness the same
            gal =   gal.expand(rescale)
            # rotate by 'ang'
            gal =   gal.rotate(ang)
            # accounting for zeropoint difference between COSMOS HST and HSC
            gal =   gal*flux_scaling
            # shear distortion
            gal =   gal.shear(g1=g1,g2=g2)
            if 'Shift' in outDir:
                # Galaxies is randomly shifted
                # This shift ensure that the offset to (ngrid//2,ngrid//2) is an isotropic circle
                dx = ud()*scale # ud from 0~1
                dy = ud()*scale
                if i==0:
                    logging.info('%.2f,%.2f' %(dx,dy))
                gal= gal.shift(dx,dy)
            elif 'Center' in outDir:
                # Galaxies is located at (ngrid//2,ngrid//2)
                dx = 0.5*scale
                dy = 0.5*scale
                gal= gal.shift(dx,dy)
            else:
                #Galaxies is located at (ngrid//2-0.5,ngrid//2-0.5))
                pass
            gal =   galsim.Convolve([psfInt,gal],gsparams=bigfft)
            # draw galaxy
            sub_img =   gal_image[b]
            gal.drawImage(sub_img,add_to_image=True)
            del gal,b,sub_img
        del inCat,cosmos_cat,cosmo252,psfInt
        gc.collect()
    elif 'small' in outDir:
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
        gc.collect()
    else:
        raise ValueError("outDir should cotain 'basic' or 'small'!!")
    del ud
    if do_write:
        gal_image.write(outFname,clobber=True)
    if return_array:
        return gal_image.array

def make_gal_ssbg(shear,psf,rng,r1,r0=20.):
    """
    simulate an exponential object with moffat PSF, given a SNR (r0) and
    a source background noise ratio (r0)

    Args:
        shear (tuple):
           (g1, g2),The shear in each component
        rng (randState):
            The random number generator
        r1  (float):
            The source background noise variance ratio
        r0  (float):
            The SNR of galaxy
        psf (galsim.Moffat):
            galsim.Moffat(beta=2.5,fwhm=psf_fwhm,).shear(g1=0.02, g2=-0.01,)

    Returns:
       img (ndarray):
            noisy image array
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
