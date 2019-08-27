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
import astropy.table as astTab

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

class analyzePrepareConfig(pexConfig.Config):
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

class analyzePrepareTask(pipeBase.CmdLineTask):
    _DefaultName = "analyzePrepare"
    ConfigClass = analyzePrepareConfig
    def __init__(self,**kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)
        self.schema     =   afwTable.SourceTable.makeMinimalSchema()
        self.makeSubtask("readDataSim",schema=self.schema)        
        self.makeSubtask('fpfsBase', schema=self.schema)
        
        
    @pipeBase.timeMethod
    def run(self,index):
        rootDir     =   self.config.rootDir
        outputdir   =   os.path.join(self.config.rootDir,'outcomeFPFS')
        if not os.path.exists(outputdir):
            self.log.info('cannot find the output directory')
            return
        g1List      =   [-0.02 ,-0.025,0.03 ,0.01,-0.008,-0.015, 0.022,0.005]
        g2List      =   [-0.015, 0.028,0.007,0.00, 0.020,-0.020,-0.005,0.010]
        for ig in range(8):
            for irot in range(4):
                prepend =   '-id%d-g%d-r%d' %(index,ig,irot)
                self.log.info('index: %d, shear: %d, rot: %d' %(index,ig,irot))
                inFname =   'src%s.fits' %(prepend)
                inFname =   os.path.join(outputdir,outFname)
                if not os.path.exists(inFname):
                    self.log.info('cannot find the output file%s' %prepend)
                    continue
                src     =   astTab.Table.read(inFname)
                src     =   self.keepUnique(src)

                
        return

    def keepUnique(self,src):
        src['ipos']     =   (src['base_SdssCentroid_y']//64)*100 +(src['base_SdssCentroid_x']//64)
        src['ipos']     =   src['ipos'].astype(np.int)
        src['centDist'] =   ((src['base_SdssCentroid_y']%64-32)**2. +
                                (src['base_SdssCentroid_x']%64-32)**2.)
        src['centDist'] =   np.sqrt(src['centDist'])
        # First, keep only detections that are the closest to the grid point
        # Get sorted index by grid index and grid distance
        inds        =   np.lexsort([src['centDist'], src['ipos']])
        inds_unique =   np.unique(src['ipos'][inds], return_index=True)[1]
        src     =   src[inds[inds_unique]]
        src     =   src[(src['centDist']<5.)]
        return src
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
        

class analyzePrepareDriverConfig(pexConfig.Config):
    perGroup=   pexConfig.Field(dtype=int, default=100, doc = 'data per field')
    analyzePrepare = pexConfig.ConfigurableField(
        target = analyzePrepareTask,
        doc = "analyzePrepare task to run on multiple cores"
    )
    def setDefaults(self):
        pexConfig.Config.setDefaults(self)

class analyzePrepareRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        minGroup    =  parsedCmd.minGroup 
        maxGroup    =  parsedCmd.maxGroup 
        return [(ref, kwargs) for ref in range(minGroup,maxGroup)] 

def unpickle(factory, args, kwargs):
    """Unpickle something by calling a factory"""
    return factory(*args, **kwargs)

class analyzePrepareDriverTask(BatchPoolTask):
    ConfigClass = analyzePrepareDriverConfig
    RunnerClass = analyzePrepareRunner
    _DefaultName = "analyzePrepareDriver"
    
    def __reduce__(self):
        """Pickler"""
        return unpickle, (self.__class__, [], dict(config=self.config, name=self._name,
                parentTask=self._parentTask, log=self.log))

    def __init__(self,**kwargs):
        BatchPoolTask.__init__(self, **kwargs)
        self.makeSubtask("analyzePrepare")
    
    @abortOnError
    def run(self,Id):
        perGroup=   self.config.perGroup
        fMin    =   perGroup*Id
        fMax    =   perGroup*(Id+1)
        #Prepare the pool
        pool    =   Pool("analyzePrepare")
        pool.cacheClear()
        fieldList=  range(fMin,fMax)
        pool.map(self.process,fieldList)
        return
        
    def process(self,cache,ifield):
        self.analyzePrepare.run(ifield)
        self.log.info('finish field %03d' %(ifield))
        return

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        kwargs.pop("doBatch", False)
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_argument('--minGroup', type= int, 
                        default=0,
                        help='minimum group number')
        parser.add_argument('--maxGroup', type= int, 
                        default=1,
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
