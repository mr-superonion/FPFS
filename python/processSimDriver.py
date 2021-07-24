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
import fpfsBase
import numpy as np
import astropy.io.fits as pyfits

# lsst Tasks
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

import lsst.obs.subaru.filterFraction
from readDataSim import readDataSimTask

class processSimConfig(pexConfig.Config):
    "config"
    doHSM   = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Whether run HSM",
    )
    doFPFS  = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Whether run FPFS",
    )
    readDataSim= pexConfig.ConfigurableField(
        target  = readDataSimTask,
        doc     = "Subtask to run measurement hsm"
    )
    rootDir     = pexConfig.Field(
        dtype=str,
        default="./",
        doc="Root Diectory"
    )
    def setDefaults(self):
        pexConfig.Config.setDefaults(self)
        self.readDataSim.doWrite=   False
        self.readDataSim.doDeblend= True
        self.readDataSim.doAddFP=   True

class processSimTask(pipeBase.CmdLineTask):
    _DefaultName=   "processSim"
    ConfigClass =   processSimConfig
    def __init__(self,**kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)
        self.schema     =   afwTable.SourceTable.makeMinimalSchema()
        self.makeSubtask("readDataSim",schema=self.schema)

    @pipeBase.timeMethod
    def runDataRef(self,ifield):
        # Basic
        rootDir     =   self.config.rootDir
        igroup      =   ifield//250
        self.log.info('running for group: %s, field: %s' %(igroup,ifield))
        nn          =   100
        ngal        =   nn*nn
        ngrid       =   64
        beta        =   0.75
        noiVar      =   7e-3
        opend       =   'var7em3'
        pixScale    =   0.168
        psfFWHM     =   '60'
        psfFWHMF    =   eval(psfFWHM)/100.
        rcut        =   16#max(min(int(psfFWHMF/pixScale*4+0.5),15),12)
        beg         =   ngrid//2-rcut
        end         =   beg+2*rcut

        # necessary directories
        galDir      =   os.path.join(self.config.rootDir,'galaxy_basic_psf%s' %psfFWHM)
        noiDir      =   os.path.join(self.config.rootDir,'noise')
        assert os.path.exists(galDir)
        assert os.path.exists(noiDir)

        # noise
        noiFname    =   os.path.join(noiDir,'noi%04d.fits' %ifield)
        # multiply by 10 since the noise has variance 0.01
        noiData     =   pyfits.getdata(noiFname)*10.
        # same for the noivar model
        powIn       =   np.load('corPre/noiPows2.npy',allow_pickle=True).item()['%s'%rcut]*noiVar*100
        powModel    =   np.zeros((1,powIn.shape[0],powIn.shape[1]))
        powModel[0] =   powIn

        # PSF
        psfFname    =   os.path.join(galDir,'psf-%s.fits' %psfFWHM)
        psfData     =   pyfits.getdata(psfFname)
        npad        =   (ngrid-psfData.shape[0])//2
        psfData2    =   np.pad(psfData,(npad+1,npad),mode='constant')
        assert psfData2.shape[0]==ngrid
        psfData2    =   psfData2[beg:end,beg:end]

        # Task
        fpTask      =   fpfsBase.fpfsTask(psfData2,noiFit=powModel[0],beta=beta)

        outDir1     =   os.path.join(self.config.rootDir,'outcome-%s' %opend,\
                'src-psf%s-%s' %(psfFWHM,igroup))
        if not os.path.isdir(outDir1):
            os.mkdir(outDir1)
        outDir2     =   os.path.join(self.config.rootDir,'outcome-%s' %opend,\
                'fpfs-rcut16-psf%s-%s' %(psfFWHM,igroup))
        if not os.path.isdir(outDir2):
            os.mkdir(outDir2)

        #isList      =   ['g1-0000','g2-0000','g1-2222','g2-2222']
        isList      =   ['g1-1111']
        for ishear in isList:
            galFname    =   os.path.join(galDir,'image-%s-%s.fits' %(igroup,ishear))
            galData     =   pyfits.getdata(galFname)+noiData*np.sqrt(noiVar)

            outFname    =   os.path.join(outDir1,'src%04d-%s.fits' %(ifield,ishear))
            if not os.path.exists(outFname) and self.config.doHSM:
                exposure    =   self.makeHSCExposure(galData,psfData,pixScale,noiVar)
                src         =   self.readDataSim.measureSource(exposure)
                wFlag       =   afwTable.SOURCE_IO_NO_FOOTPRINTS
                src.writeFits(outFname,flags=wFlag)
                del exposure,src
                gc.collect()
            else:
                self.log.info('Skipping HSM measurement: %04d, %s' %(ifield,ishear))

            outFname    =   os.path.join(outDir2,'src%04d-%s.fits' %(ifield,ishear))
            if not os.path.exists(outFname) and self.config.doFPFS:
                imgList=[galData[i//nn*ngrid+beg:i//nn*ngrid+end,i%nn*ngrid+beg:i%nn*ngrid+end] for i in range(ngal)]
                out=fpTask.measure(imgList)
                pyfits.writeto(outFname,out)
                del out,imgList
                gc.collect()
            else:
                self.log.info('Skipping FPFS measurement: %04d, %s' %(ifield,ishear))

            del galData,outFname
            gc.collect()
        return

    def makeHSCExposure(self,galData,psfData,pixScale,variance):
        ny,nx=galData.shape
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
        cdelt   =   (pixScale*afwGeom.arcseconds)
        CD      =   afwGeom.makeCdMatrix(cdelt, afwGeom.Angle(0.))#no rotation
        #wcs
        crval   =   afwGeom.SpherePoint(afwGeom.Angle(0.,afwGeom.degrees),afwGeom.Angle(0.,afwGeom.degrees))
        #crval   =   afwCoord.IcrsCoord(0.*afwGeom.degrees, 0.*afwGeom.degrees) # hscpipe6
        crpix   =   afwGeom.Point2D(0.0, 0.0)
        dataWcs =   afwGeom.makeSkyWcs(crpix,crval,CD)
        exposure.setWcs(dataWcs)
        #prepare the frc
        dataCalib = afwImg.makePhotoCalibFromCalibZeroPoint(63095734448.0194)
        exposure.setPhotoCalib(dataCalib)
        return exposure

    @classmethod
    def _makeArgumentParser(cls):
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        return parser
    @classmethod
    def writeConfig(self, butler, clobber=False, doBackup=False):
        pass
    def writeSchemas(self, butler, clobber=False, doBackup=False):
        pass
    def writeMetadata(self, ifield):
        pass
    def writeEupsVersions(self, butler, clobber=False, doBackup=False):
        pass


class processSimDriverConfig(pexConfig.Config):
    processSim = pexConfig.ConfigurableField(
        target = processSimTask,
        doc = "processSim task to run on multiple cores"
    )
    def setDefaults(self):
        pexConfig.Config.setDefaults(self)

class processSimRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        minIndex    =  parsedCmd.minIndex
        maxIndex    =  parsedCmd.maxIndex
        return [(ref, kwargs) for ref in range(minIndex,maxIndex)]

def unpickle(factory, args, kwargs):
    """Unpickle something by calling a factory"""
    return factory(*args, **kwargs)

class processSimDriverTask(BatchPoolTask):
    ConfigClass = processSimDriverConfig
    RunnerClass = processSimRunner
    _DefaultName = "processSimDriver"

    def __reduce__(self):
        """Pickler"""
        return unpickle, (self.__class__, [], dict(config=self.config, name=self._name,
                parentTask=self._parentTask, log=self.log))

    def __init__(self,**kwargs):
        BatchPoolTask.__init__(self, **kwargs)
        self.makeSubtask("processSim")

    @abortOnError
    def runDataRef(self,index):
        #Prepare the pool
        pool    =   Pool("processSim")
        pool.cacheClear()
        fieldList=np.arange(100*index,100*(index+1))
        pool.map(self.process,fieldList)
        return

    def process(self,cache,prepend):
        self.processSim.runDataRef(prepend)
        self.log.info('finish %s' %(prepend))
        return

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        kwargs.pop("doBatch", False)
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_argument('--minIndex', type= int,
                        default=0,
                        help='minimum Index number')
        parser.add_argument('--maxIndex', type= int,
                        default=1,
                        help='maximum Index number')
        return parser

    @classmethod
    def batchWallTime(cls, time, parsedCmd, numCpus):
        return None
    def writeConfig(self, butler, clobber=False, doBackup=False):
        pass
    def writeSchemas(self, butler, clobber=False, doBackup=False):
        pass
    def writeMetadata(self, ifield):
        pass
    def writeEupsVersions(self, butler, clobber=False, doBackup=False):
        pass
    def _getConfigName(self):
        return None
    def _getEupsVersionsName(self):
        return None
    def _getMetadataName(self):
        return None
