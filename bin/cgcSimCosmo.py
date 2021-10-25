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
import gc
import galsim
import fitsio
import imgSimutil
import numpy as np
from pixel3D import cartesianGrid3D
from configparser import ConfigParser

# lsst Tasks
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

from lsst.pipe.base import TaskRunner
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, abortOnError

class cgcSimCosmoBatchConfig(pexConfig.Config):
    def setDefaults(self):
        pexConfig.Config.setDefaults(self)
    def validate(self):
        pexConfig.Config.validate(self)
class cgcSimCosmoRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        #Prepare the pool
        p2List  =   ['0000','2222','2000','0200','0020','0002']
        #p2List  =   ['0000','1111','2222','2000','0200','0020','0002','2111','1211','1121','1112']
        p1List  =   ['g1'] #['g1','g2']
        pendList=   ['%s-%s' %(i1,i2) for i1 in p1List for i2 in p2List]
        return [(ref, kwargs) for ref in pendList]
def unpickle(factory, args, kwargs):
    """Unpickle something by calling a factory"""
    return factory(*args, **kwargs)
class cgcSimCosmoBatchTask(BatchPoolTask):
    ConfigClass = cgcSimCosmoBatchConfig
    RunnerClass = cgcSimCosmoRunner
    _DefaultName = "cgcSimCosmoBatch"
    def __reduce__(self):
        """Pickler"""
        return unpickle, (self.__class__, [], dict(config=self.config, name=self._name,
                parentTask=self._parentTask, log=self.log))
    def __init__(self,**kwargs):
        BatchPoolTask.__init__(self, **kwargs)
        return

    @abortOnError
    def runDataRef(self,pend):
        self.log.info('begining for setup %s' %(pend))
        #Prepare the storeSet
        pool    =   Pool("cgcSimCosmoBatch")
        pool.cacheClear()
        expDir  =   "galaxy_cosmoE_psf75"
        if not os.path.isdir(expDir):
            os.mkdir(expDir)
        pool.storeSet(expDir=expDir)
        pool.storeSet(pend=pend)

        hpList  =  imgSimutil.cosmoHSThpix
        pool.map(self.process,hpList)
        self.log.info('finish setup %s' %(pend))
        return

    @abortOnError
    def process(self,cache,pixId):

        pend   =   cache.pend
        psfFWHM =   eval(cache.expDir.split('_psf')[-1])/100.
        outFname=   os.path.join(cache.expDir,'image-%d-%s.fits' %(pixId,pend))

        if os.path.isfile(outFname):
            self.log.info('Already have the outcome')
        else:
            self.processExp(pixId,pend,psfFWHM,outFname)
        return

    def processExp(self,pixId,pend,psfFWHM,outFname):
        # Galsim galaxies
        directory   =   os.path.join(os.environ['homeWrk'],'COSMOS/galsim_train/COSMOS_25.2_training_sample/')
        catName     =   'real_galaxy_catalog_25.2.fits'
        cosmos_cat  =   galsim.COSMOSCatalog(catName,dir=directory)

        # Basic parameters
        scale       =   0.168
        bigfft      =   galsim.GSParams(maximum_fft_size=10240)
        flux_scaling=   2.587

        # Get the shear information
        # Three choice on g(-0.02,0,0.02)
        gList   =   np.array([-0.02,0.,0.02])
        gList   =   gList[[eval(i) for i in pend.split('-')[-1]]]
        self.log.info('Processing for pixId %d' %pixId)
        self.log.info('shear List is for %s' %gList)

        # PSF
        self.log.info('The FHWM for PSF is: %s arcsec'%psfFWHM)
        psfInt  =   galsim.Moffat(beta=3.5,fwhm=psfFWHM,trunc=psfFWHM*4.)
        psfInt  =   psfInt.shear(e1=0.02,e2=-0.02)

        pix_scale=  0.168
        # catalog
        trainName=  '252E' # 252
        cosmo252=   imgSimutil.cosmoHSTGal(trainName)
        dd      =   cosmo252.hpInfo[cosmo252.hpInfo['pix']==pixId]
        nx      =   int(dd['dra']/pix_scale*3600.)+1
        ny      =   int(dd['ddec']/pix_scale*3600.)+1
        hscCat  =   cosmo252.readHpixSample(pixId)
        msk     =   (hscCat['xI']>32)&(hscCat['yI']>32)\
                &(hscCat['xI']<nx-32)&(hscCat['yI']<ny-32)
        hscCat  =   hscCat[msk]
        del msk
        self.log.info('total %d galaxies' %(len(hscCat)))
        zbound  =   [0.,0.561,0.906,1.374,5.410]
        gal_image   =   galsim.ImageF(nx,ny,scale=scale)
        gal_image.setOrigin(0,0)

        for ss  in hscCat:
            if pend.split('-')[0]=='g1':
                g1=gList[np.where((ss['zphot']>zbound[:-1])&(ss['zphot']<=zbound[1:]))[0]][0]
                g2=0.
            elif pend.split('-')[0]=='g2':
                g1=0.
                g2=gList[np.where((ss['zphot']>zbound[:-1])&(ss['zphot']<=zbound[1:]))[0]][0]
            # each galaxy
            gal =   cosmos_cat.makeGalaxy(gal_type='parametric',index=ss['index'],gsparams=bigfft)
            flux=   10**((27.-ss['mag_auto'])/2.5)
            #self.log.info('%s' %(flux/gal.flux))
            gal.withFlux(flux)
            #gal =   gal*flux_scaling
            gal =   gal.shear(g1=g1,g2=g2)
            gal =   gal.shift(ss['dx'],ss['dy'])
            gal =   galsim.Convolve([psfInt,gal],gsparams=bigfft)
            # draw galaxy
            b   =   galsim.BoundsI(ss['xI']-32,ss['xI']+32,ss['yI']-32,ss['yI']+32)
            sub_img =   gal_image[b]
            gal.drawImage(sub_img,add_to_image=True)
            del gal,b,sub_img
            gc.collect()
        gal_image.write(outFname,clobber=True)
        del hscCat,cosmos_cat,cosmo252,psfInt
        gc.collect()
        return

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        kwargs.pop("doBatch", False)
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
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
