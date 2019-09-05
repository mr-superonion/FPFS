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
from configparser import ConfigParser

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



class cgcSimTask(pipeBase.CmdLineTask):
    _DefaultName=   "cgcSim"
    ConfigClass =   pexConfig.Config
    def __init__(self,**kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)

    
    @pipeBase.timeMethod
    def run(self,index,rootDir,conpend):
        rootDir2    =   os.path.join(rootDir,conpend)
        configName  =   'noiPsfConfig/config_%s.ini' %conpend
        self.log.info('using configuration %s' %(configName))
        self.log.info('simulating index %s' %(index))
        parser      =   ConfigParser()
        parser.read(configName)
        # Basic parameters
        ngrid       =   64
        nx          =   50 
        ny          =   nx
        ndata       =   nx*ny
        nrot        =   4
        scale       =   0.168
        ngridTot    =   ngrid*nx
        bigfft      =   galsim.GSParams(maximum_fft_size=10240)
        flux_scaling=   2.587
        variance    =   parser.getfloat('noise','variance')
        corFname    =   parser.get('noise','corFname')
        g1List      =   [-0.02 ,-0.025,0.03 ,0.01,-0.008,-0.015, 0.022,0.005]
        g2List      =   [-0.015, 0.028,0.007,0.00, 0.020,-0.020,-0.005,0.010]
        
        # Get the psf and nosie information 
        # Get the PSF image
        psf_beta    =   3.5 
        psf_fwhm    =   parser.getfloat('psf','fwhm') #arcsec
        psf_trunc   =   5.*psf_fwhm # arcsec
        psf_e1      =   0.          #
        psf_e2      =   0.025       #
        psf         =   galsim.Moffat(beta=psf_beta, 
                        fwhm=psf_fwhm,trunc=psf_trunc)
        psf         =   psf.shear(e1=psf_e1,e2=psf_e2)        
        psfImg      =   psf.drawImage(nx=45,ny=45,scale=scale)
        # Get the  galaxy generator      
        flux_scaling=   2.587*15
        # Load data
        catName     =   'real_galaxy_catalog_25.2.fits'
        dir         =   '../galsim_train/COSMOS_25.2_training_sample/'
        cosmos_cat  =   galsim.COSMOSCatalog(catName, dir=dir)
        # index
        index_use   =   cosmos_cat.orig_index
        # parametric catalog
        param_cat   =   cosmos_cat.param_cat[index_use]
        record      =   param_cat[index]
        # prepare the galaxies        
        gal0        =   cosmos_cat.makeGalaxy(gal_type='parametric',index=index,gsparams=bigfft)
        gal0        *=  flux_scaling
        ud          =   galsim.UniformDeviate(index*10000+1)
        # rotate the galaxy
        ang         =   ud()*2.*np.pi * galsim.radians
        gal0        =   gal0.rotate(ang)
        for irot in range(4):
            angR=   np.pi/4.*irot*galsim.radians
            galR=   gal0.rotate(angR)
            for ig in range(8):
                prepend     =   '-id%d-g%d-r%d'%(index,ig,irot)
                outFname    =   os.path.join(rootDir2,'expSim','image%s.fits' %prepend)
                if os.path.exists(outFname):
                    exposure=   afwImg.ExposureF.readFits(outFname)
                    exposure.getMaskedImage().getVariance().getArray()[:,:]=variance
                    exposure.writeFits(outFname)
                    self.log.info('Already have the outcome for %s' %prepend)
                    continue
                g1  =   g1List[ig]
                g2  =   g2List[ig]
                # Shear the galaxy
                gal     =   galR.shear(g1=g1,g2=g2)
                final   =   galsim.Convolve([psf,gal],gsparams=bigfft)
                # setup the galaxy image and the noise image
                gal_image   =   galsim.ImageF(nx*ngrid,ny*ngrid,scale=scale)
                gal_image.setOrigin(0,0)
                var_image   =   galsim.ImageF(nx*ngrid,ny*ngrid,scale=scale)
                var_image.setOrigin(0,0)
                i           =   0
                while i <ndata:
                    # Prepare the subimage
                    ix      =   i%nx
                    iy      =   i//nx
                    b       =   galsim.BoundsI(ix*ngrid,(ix+1)*ngrid-1,iy*ngrid,(iy+1)*ngrid-1)
                    sub_gal_image = gal_image[b]
                    # Draw the galaxy image
                    final.drawImage(sub_gal_image,method='no_pixel')
                    i       +=  1
                self.log.info('Adding correlated noise')
                rng         =   galsim.BaseDeviate(index*32+ig+irot*8)
                corNoise    =   galsim.getCOSMOSNoise(file_name=corFname,rng=rng,cosmos_scale=scale,variance=variance)
                corNoise.applyTo(gal_image)
                #Make Lsst exposure
                exposure    =   afwImg.ExposureF(nx*ngrid,ny*ngrid)
                exposure.getMaskedImage().getImage().getArray()[:,:]=gal_image.array
                exposure.getMaskedImage().getVariance().getArray()[:,:]=variance
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

class cgcSimBatchConfig(pexConfig.Config):
    perGroup=   pexConfig.Field(dtype=int, default=100, doc = 'data per field')
    cgcSim  =   pexConfig.ConfigurableField(
        target = cgcSimTask,
        doc = "cgcSim task to run on multiple cores"
    )
    rootDir =   pexConfig.Field(dtype=str, 
                default='cgc', doc = 'root directory'
    )
    conpend =   pexConfig.Field(dtype=str,
                default='fwhm4_var4', doc = 'prepend for one configuration'
    )
    def setDefaults(self):
        pexConfig.Config.setDefaults(self)
    
    def validate(self):
        pexConfig.Config.validate(self)
        if not os.path.exists(self.rootDir):
            os.mkdir(self.rootDir)
        rootDir2=   os.path.join(self.rootDir,self.conpend)
        if not os.path.exists(rootDir2):
            os.mkdir(rootDir2)
        expDir  =   os.path.join(rootDir2,'expSim')
        if not os.path.exists(expDir):
            os.mkdir(expDir)
    
class cgcSimRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        minGroup    =  parsedCmd.minGroup 
        maxGroup    =  parsedCmd.maxGroup 
        return [(ref, kwargs) for ref in range(minGroup,maxGroup)] 

def unpickle(factory, args, kwargs):
    """Unpickle something by calling a factory"""
    return factory(*args, **kwargs)

class cgcSimBatchTask(BatchPoolTask):
    ConfigClass = cgcSimBatchConfig
    RunnerClass = cgcSimRunner
    _DefaultName = "cgcSimBatch"

    def __reduce__(self):
        """Pickler"""
        return unpickle, (self.__class__, [], dict(config=self.config, name=self._name,
                parentTask=self._parentTask, log=self.log))

    def __init__(self,**kwargs):
        BatchPoolTask.__init__(self, **kwargs)
        self.makeSubtask("cgcSim")
    
    @abortOnError
    def run(self,Id):
        self.log.info('beginning group %d' %(Id))
        perGroup=   self.config.perGroup
        fMin    =   perGroup*Id
        fMax    =   perGroup*(Id+1)
        #Prepare the pool
        pool    =   Pool("cgcSim")
        pool.cacheClear()
        pool.storeSet(rootDir=self.config.rootDir)
        pool.storeSet(conpend=self.config.conpend)
        fieldList=  range(fMin,fMax)
        pool.map(self.process,fieldList)
        self.log.info('finish group %d'%(Id) )
        return
        
    def process(self,cache,index):
        self.cgcSim.run(index,cache.rootDir,cache.conpend)
        self.log.info('finish index %04d' %(index))
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
