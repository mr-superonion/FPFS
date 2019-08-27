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
import numpy as np
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

from lsst.pipe.base import TaskRunner
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, abortOnError

import lsst.obs.subaru.filterFraction
from fpfsBase import fpfsBaseTask
from readDataSim import readDataSimTask

class processSimConfig(pexConfig.Config):
    "config"
    readDataSim= pexConfig.ConfigurableField(
        target  = readDataSimTask,
        doc     = "Subtask to run measurement of fpfs method"
    )
    fpfsBase = pexConfig.ConfigurableField(
        target = fpfsBaseTask,
        doc = "Subtask to run measurement of fpfs method"
    )
    rootDir     = pexConfig.Field(
        dtype=str, 
        default="rgc/fwhm4_var4/", 
        doc="Root Diectory"
    )
    def setDefaults(self):
        pexConfig.Config.setDefaults(self)
        self.readDataSim.rootDir=   self.rootDir
        self.readDataSim.doWrite=   False
        self.fpfsBase.doTest    =   False
        self.fpfsBase.doFD      =   False
        self.fpfsBase.dedge     =   2

class processSimTask(pipeBase.CmdLineTask):
    _DefaultName = "processSim"
    ConfigClass = processSimConfig
    def __init__(self,**kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)
        self.schema     =   afwTable.SourceTable.makeMinimalSchema()
        self.makeSubtask("readDataSim",schema=self.schema)        
        self.makeSubtask('fpfsBase', schema=self.schema)
        
        
    @pipeBase.timeMethod
    def run(self,index):
        rootDir     =   self.config.rootDir
        inputdir    =   os.path.join(self.config.rootDir,'expSim')
        outputdir   =   os.path.join(self.config.rootDir,'outcomeFPFS')
        if not os.path.exists(outputdir):
            os.mkdir(outputdir)
        for ig in range(8):
            for irot in range(4):
                prepend     =   '-id%d-g%d-r%d' %(index,ig,irot)
                self.log.info('index: %d, shear: %d, rot: %d' %(index,ig,irot))
                inFname     =   os.path.join(inputdir,'image%s.fits' %(prepend))
                if not os.path.exists(inFname):
                    self.log.info('Cannot find the input exposure')
                    return
                outFname    =   'src%s.fits' %(prepend)
                outFname    =   os.path.join(outputdir,outFname)
                if os.path.exists(outFname):
                    self.log.info('Already have the output file%s' %prepend)
                    continue
                dataStruct  =   self.readDataSim.readData(prepend)
                if dataStruct is None:
                    self.log.info('failed to read data')
                    continue
                else:
                    self.log.info('successed in reading data')
                self.fpfsBase.run(dataStruct)
                dataStruct.sources.writeFits(outFname,flags=afwTable.SOURCE_IO_NO_FOOTPRINTS)
        return
    
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
    perGroup=   pexConfig.Field(dtype=int, default=100, doc = 'data per field')
    processSim = pexConfig.ConfigurableField(
        target = processSimTask,
        doc = "processSim task to run on multiple cores"
    )
    def setDefaults(self):
        pexConfig.Config.setDefaults(self)

class processSimRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        minGroup    =  parsedCmd.minGroup 
        maxGroup    =  parsedCmd.maxGroup 
        return [(ref, kwargs) for ref in range(minGroup,maxGroup)] 

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
    def run(self,Id):
        perGroup=   self.config.perGroup
        fMin    =   perGroup*Id
        fMax    =   perGroup*(Id+1)
        #Prepare the pool
        pool    =   Pool("processSim")
        pool.cacheClear()
        fieldList=  range(fMin,fMax)
        pool.map(self.processSim.run,fieldList)
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
