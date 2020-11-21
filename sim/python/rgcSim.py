#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 20082014 LSST Corporation.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
# python lib
import os
import logging
import galsim
import numpy as np
from astropy.table import Table
import astropy.io.fits as pyfits

# lsst Tasks
import lsst.daf.base as dafBase
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.afw.image as afwImg
import lsst.afw.detection as afwDet
import lsst.afw.geom as afwGeom
import lsst.afw.coord as afwCoord
import lsst.meas.algorithms as meaAlg

from lsst.pipe.base import TaskRunner
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, abortOnError


class rgcSimConfig(pexConfig.Config):
    expDir      =   pexConfig.Field(dtype=str, default='expDir', doc = 'directory to store exposures')
    catPre      =   pexConfig.Field(dtype=str, default='catPre', doc = 'directory to store pre-catalog')
    psfPre      =   pexConfig.Field(dtype=str, default='psfPre', doc = 'directory to store pre-PSF')
    def setDefaults(self):
        pexConfig.Config.setDefaults(self)
    
    def validate(self):
        pexConfig.Config.validate(self)
        if not os.path.exists(self.expDir):
            os.mkdir(self.expDir)
        if not os.path.exists(self.catPre):
            os.mkdir(self.catPre)
        if not os.path.exists(self.psfPre):
            raise RuntimeError("Cannot find PSF images")

class rgcSimTask(pipeBase.CmdLineTask):
    _DefaultName = "rgcSim"
    ConfigClass = rgcSimConfig
    def __init__(self,**kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)

    
    @pipeBase.timeMethod
    def run(self,ifield):
        badIDENT    =   [78798,276775,908556,81452,200743]
        self.log.info('begining for field %04d' %(ifield))
        outFname    =   os.path.join(self.config.expDir,'image-%04d.fits' %(ifield))
        if os.path.exists(outFname):
            self.log.info('Already have the outcome')
            return
        self.log.info('simulating field %s' %(ifield))
        # Basic parameters
        ngrid       =   64
        nx          =   100 
        ny          =   nx
        ndata       =   nx*ny
        nrot        =   2
        scale       =   0.168
        ngridTot    =   ngrid*nx
        bigfft      =   galsim.GSParams(maximum_fft_size=10240)
        flux_scaling=   2.587
        
        # Get the psf and nosie information 
        psfFname    =   os.path.join(self.config.psfPre,'psf%04d.fits'%(ifield))
        psfImg      =   galsim.fits.read(psfFname)
        noiseHead   =   pyfits.getheader(psfFname)
        psfInt      =   galsim.InterpolatedImage(psfImg,scale=scale,flux = 1.) 
        # Get the shear information
        g1          =   noiseHead['g1']
        g2          =   noiseHead['g2']
        fwhm        =   noiseHead['fwhm'] 
        if fwhm <0.6 and fwhm>0.2:
            iparent =   0
        elif fwhm<=0.85 and fwhm>=0.6:
            iparent =   0
        elif fwhm>0.85:
            iparent =   0
        else:
            self.log.info('incorrect PSF fwhm')
            return
        # Get the  galaxy generator      
        # Load data
        Xnames      =   ['best','median','worst']
        Xname       =   Xnames[iparent]
        catName     =   'real_galaxy_catalog_%s.fits' %(Xname)
        directory   =   '/work/xiangchong.li/work/galsim_train/parent_%s_processed' %(Xname)
        rgc         =   galsim.RealGalaxyCatalog(catName,dir=directory)
        rgcCat      =   rgc.cat
        nrgcDat     =   len(rgcCat)
        if iparent  ==  1: 
            badIdList   =   [148999,100999] 
        else:
            badIdList   =   []
        extFname    =   '/work/xiangchong.li/work/galsim_train/cosmos_extinction_HSC.hdf5'
        extTab      =   Table.read(extFname,"/%s" %iparent)
        assert  len(extTab)==nrgcDat, 'number of galaxies in extinction file does not math that of rgc catalog'

        variance    =   noiseHead['var']
        rescale     =   noiseHead['rescale']
        ud          =   galsim.UniformDeviate(ifield*10000+1)
        np.random.seed(ifield*10000+1)

        # setup the galaxy image and the noise image
        gal_image   =   galsim.ImageF(nx*ngrid,ny*ngrid,scale=scale)
        gal_image.setOrigin(0,0)
        var_image   =   galsim.ImageF(nx*ngrid,ny*ngrid,scale=scale)
        var_image.setOrigin(0,0)
        data_rows   =   []
        i           =   0
        while i <ndata:
            #self.log.info('processing stamp %d of field %d' %(i,ifield))
            # Prepare the subimage
            ix      =   i%nx
            iy      =   i//nx
            b       =   galsim.BoundsI(ix*ngrid,(ix+1)*ngrid-1,iy*ngrid,(iy+1)*ngrid-1)
            sub_gal_image = gal_image[b]
            sub_var_image = var_image[b]
            #simulate the galaxy
            if i%nrot==0:
                while True:
                    index   =   np.random.randint(0,nrgcDat)
                    if (index not in badIdList) and (rgcCat[index]['IDENT'] not in badIDENT):
                        break
                gal0    =   galsim.RealGalaxy(rgc,index=index,gsparams=bigfft)
                gal0    *=  flux_scaling
                # rotate the galaxy
                ang     =   ud()*2.*np.pi * galsim.radians
                gal0    =   gal0.rotate(ang)
            else:
                gal0    =   gal0.rotate(1./nrot*np.pi*galsim.radians)
            ss      =   rgcCat[index]
            ssExt   =   extTab[index]
            assert ssExt['IDENT']==ss['IDENT'], 'IDENT does not match'
            # Shear the galaxy
            gal     =   gal0.shear(g1=g1,g2=g2)
            final   =   galsim.Convolve([psfInt,gal],gsparams=bigfft)
            # Draw the galaxy image
            final.drawImage(sub_gal_image,method='no_pixel')
            #whiten the noise
            galNoiVar   =   sub_gal_image.whitenNoise(final.noise)
            sub_var_image+=  galNoiVar
            row=(i,index,ss['IDENT'],ss['RA'],ss['DEC'],ss['MAG'],ssExt['a_i'],iparent)
            data_rows.append(row)
            i   +=  1
        self.log.info('Adding correlated noise')
        rng = galsim.BaseDeviate(ifield)
        max_variance=   np.max(var_image.array)
        var_image   =   max_variance - var_image
        vn          =   galsim.VariableGaussianNoise(rng,var_image)
        gal_image.addNoise(vn)
        corNoise    =   galsim.getCOSMOSNoise(file_name='./corPre/correlation.fits',rng=rng,cosmos_scale=scale,variance=variance)
        unCorNoise  =   galsim.UncorrelatedNoise(max_variance,rng=rng,scale=scale)
        corNoise    =   corNoise-unCorNoise
        corNoise.applyTo(gal_image)
        exposure    =   afwImg.ExposureF(nx*ngrid,ny*ngrid)
        exposure.getMaskedImage().getImage().getArray()[:,:]=gal_image.array
        exposure.getMaskedImage().getVariance().getArray()[:,:]=variance*rescale
        del gal_image
        del var_image
        #Set the PSF
        psfArray    =   psfImg.array
        ngridPsf    =   psfArray.shape[0]
        psfLsst     =   afwImg.ImageF(ngridPsf,ngridPsf)
        psfLsst.getArray()[:,:]= psfArray
        psfLsst     =   psfLsst.convertD()
        kernel      =   afwMath.FixedKernel(psfLsst)
        kernelPSF   =   meaAlg.KernelPsf(kernel)
        exposure.setPsf(kernelPSF)
        #prepare the wcs
        #Rotation
        cdelt   =   (0.168*afwGeom.arcseconds)
        CD      =   afwGeom.makeCdMatrix(cdelt, afwGeom.Angle(0.))#no rotation
        #wcs
        crval   =   afwCoord.IcrsCoord(0.*afwGeom.degrees, 0.*afwGeom.degrees)
        crpix   =   afwGeom.Point2D(0.0, 0.0)
        dataWcs =   afwGeom.makeSkyWcs(crpix,crval,CD)
        exposure.setWcs(dataWcs)
        #prepare the frc
        dataCalib = afwImg.Calib()
        dataCalib.setFluxMag0(63095734448.0194)
        exposure.setCalib(dataCalib)
        self.log.info('writing exposure')
        exposure.writeFits(outFname)
        del exposure
        t = Table(rows=data_rows, names=('ipos','COSMOS_INDEX','IDENT','RA','DEC','COSMOS_MAG','a_i','parent_seeing'))
        catPreName  =   os.path.join(self.config.catPre,'catalog-%04d.fits' %(ifield))
        t.write(catPreName,overwrite=True)
        del t
        return

    @classmethod
    def _makeArgumentParser(cls):
        """Create an argument parser
        """
        doBatch = kwargs.pop("doBatch", False)
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        return parser

    def writeConfig(self, butler, clobber=False, doBackup=False):
        pass

    def writeSchemas(self, butler, clobber=False, doBackup=False):
        pass

    def writeMetadata(self, dataRef):
        pass

    def writeEupsVersions(self, butler, clobber=False, doBackup=False):
        pass

class rgcSimBatchConfig(pexConfig.Config):
    perGroup =   pexConfig.Field(dtype=int, default=80, doc = 'data per field')
    rgcSim = pexConfig.ConfigurableField(
        target = rgcSimTask,
        doc = "rgcSim task to run on multiple cores"
    )
    
class rgcSimRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        minGroup    =  parsedCmd.minGroup 
        maxGroup    =  parsedCmd.maxGroup 
        return [(ref, kwargs) for ref in range(minGroup,maxGroup)] 

def unpickle(factory, args, kwargs):
    """Unpickle something by calling a factory"""
    return factory(*args, **kwargs)

class rgcSimBatchTask(BatchPoolTask):
    ConfigClass = rgcSimBatchConfig
    RunnerClass = rgcSimRunner
    _DefaultName = "rgcSimBatch"

    def __reduce__(self):
        """Pickler"""
        return unpickle, (self.__class__, [], dict(config=self.config, name=self._name,
                parentTask=self._parentTask, log=self.log))

    def __init__(self,**kwargs):
        BatchPoolTask.__init__(self, **kwargs)
        self.makeSubtask("rgcSim")
    
    @abortOnError
    def run(self,Id):
        self.log.info('beginning group %d' %(Id))
        perGroup=   self.config.perGroup
        fMin    =   perGroup*Id
        fMax    =   perGroup*(Id+1)
        #Prepare the pool
        pool    =   Pool("rgcSim")
        pool.cacheClear()
        fieldList=  range(fMin,fMax)
        pool.map(self.process,fieldList)
        self.log.info('finish group %d'%(Id) )
        return
        
    def process(self,cache,ifield):
        self.rgcSim.run(ifield)
        try:
            self.log.info('finish field %04d' %(ifield))
        except:
            self.log.info('fail to finish field %04d'%(ifield) )
        return 

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        kwargs.pop("doBatch", False)
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_argument('--minGroup', type= int, 
                        default=0,
                        help='minimum group number')
        parser.add_argument('--maxGroup', type= int, 
                        default=10,
                        help='maximum group number')
        return parser
    
    @classmethod
    def writeConfig(self, butler, clobber=False, doBackup=False):
        pass

    def writeSchemas(self, butler, clobber=False, doBackup=False):
        pass
    
    def writeMetadata(self, dataRef):
        pass
    
    def writeEupsVersions(self, butler, clobber=False, doBackup=False):
        pass
    
    def _getConfigName(self):
        return None
   
    def _getEupsVersionsName(self):
        return None
    
    def _getMetadataName(self):
        return None
