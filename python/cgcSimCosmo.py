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
    perGroup =   pexConfig.Field(dtype=int, default=100, doc = 'data per field')
    expDir      =   pexConfig.Field(dtype=str, default='galImgCosmo', doc = 'directory to store exposures')
    cgcSimCosmo = pexConfig.ConfigurableField(
        target = cgcSimCosmoTask,
        doc = "cgcSimCosmo task to run on multiple cores"
    )
    def setDefaults(self):
        pexConfig.Config.setDefaults(self)
    def validate(self):
        pexConfig.Config.validate(self)
        if not os.path.exists(self.expDir):
            os.mkdir(self.expDir)

class cgcSimCosmoRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        return [(ref, kwargs) for ref in range(1)]
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
        self.makeSubtask("cgcSimCosmo")
        return

    @abortOnError
    def runDataRef(self,Id):
        #Prepare the pool
        pool    =   Pool("cgcSimCosmo")
        pool.cacheClear()
        pool.storeSet(expDir=expDir)
        fieldList=  imgSimutil.cosmoHSThpix[-10:-9]
        pool.map(self.process,fieldList)
        return

    def process(self,cache,ihpix):
        self.log.info('begining for healPIX %d' %(ihpix))
        outFname    =   os.path.join(cache.expDir,'image-%05d-g1-2222.fits' %(ihpix))
        """
        outFname    =   os.path.join(cache.expDir,'image-%05d-g1-0000.fits' %(ihpix))
        outFname    =   os.path.join(cache.expDir,'image-%05d-g1-1111.fits' %(ihpix))
        outFname    =   os.path.join(cache.expDir,'image-%05d-g1-2000.fits' %(ihpix))
        outFname    =   os.path.join(cache.expDir,'image-%05d-g1-0200.fits' %(ihpix))
        outFname    =   os.path.join(cache.expDir,'image-%05d-g1-0020.fits' %(ihpix))
        outFname    =   os.path.join(cache.expDir,'image-%05d-g1-0002.fits' %(ihpix))
        """
        if os.path.exists(outFname):
            self.log.info('Already have the outcome')
            return

        directory   =   os.path.join(os.environ['homeWrk'],'COSMOS/galsim_train/COSMOS_25.2_training_sample/')
        catName     =   'real_galaxy_catalog_25.2.fits'
        cosmos_cat  =   galsim.COSMOSCatalog(catName,dir=directory)

        # Basic parameters
        scale       =   0.168
        bigfft      =   galsim.GSParams(maximum_fft_size=10240)
        flux_scaling=   2.587
        #varRatio   =   1.25

        # Get the shear information
        g1          =   0.02
        g2          =   0.

        # PSF
        psfInt  =   galsim.Moffat(beta=3.5,fwhm=0.65,trunc=0.65*4.)
        psfInt  =   psfInt.shear(e1=0.02,e2=-0.02)
        psfImg  =   psfInt.drawImage(nx=45,ny=45,scale=scale)

        # Stamp
        configName  =   'config-nl1.ini'
        parser  =   ConfigParser()
        parser.read(configName)
        gridInfo=   cartesianGrid3D(parser)

        # catalog
        nside   =   512
        cosmo252=   imgSimutil.cosmoHSTGal('252')
        dd      =   cosmo252.hpInfo[cosmo252.hpInfo['pix']==pixId]
        nx      =   int(dd['dra']/gridInfo.delta)
        ny      =   int(dd['ddec']/gridInfo.delta)
        hscCat  =   cosmo252.readHpixSample(pixId)
        gal_image   =   galsim.ImageF(nx,ny,scale=scale)
        gal_image.setOrigin(0,0)
        for ss  in hscCat:
            if ss['xI']-32>0 and ss['xI']+32<nx and ss['i']-32>0 and ss['yI']+32<ny:
                gal =   cosmos_cat.makeGalaxy(gal_type='parametric',index=iid,gsparams=bigfft)
                gal =   gal*flux_scaling
                gal =   gal.shear(g1=g1,g2=g2)
                gal =   galsim.Convolve([psfInt,gal],gsparams=bigfft)

                b   =   galsim.BoundsI(xI[i]-32,xI[i]+32,yI[i]-32,yI[i]+32)
                sub_img =   gal_image[b]
                gal.drawImage(sub_img,add_to_image=True)
                del gal,b,sub_img
                gc.collect()
        self.log.info('finish healPIX %d' %(ihpix))
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
