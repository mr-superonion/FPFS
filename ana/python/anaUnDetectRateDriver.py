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
import k3match
import catStat
import numpy as np

import astropy.table as astTab
import astropy.io.ascii as ascii
import anaUtil

# lsst Tasks
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base import TaskRunner
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, abortOnError

class anaUnDetectRateConfig(pexConfig.Config):
    "config"
    rootDir     = pexConfig.Field(
        dtype=str, 
        default="cgc-control-2gal/", 
        doc="Root Diectory"
    )
    def setDefaults(self):
        pexConfig.Config.setDefaults(self)

class anaUnDetectRateTask(pipeBase.CmdLineTask):
    _DefaultName = "anaUnDetectRate"
    ConfigClass = anaUnDetectRateConfig
    def __init__(self,**kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)
        
        
    @pipeBase.timeMethod
    def run(self,index):
        catPreName  =   os.path.join('catPre','control_cat.csv')
        assert os.path.exists(catPreName),'cannt find the preCat'
        cat         =   ascii.read(catPreName)[index]
        inputDir    =   os.path.join(self.config.rootDir,'outcomeFPFS')
        if not os.path.exists(inputDir):
            self.log.info('cannot find the input directory')
            return
        if min(cat['flux1'],cat['flux2'])/cat['varNoi']<500.:
            nrot    =   8
        else:
            nrot    =   4
        nshear      =   8
        g1List      =   [-0.02 ,-0.025,0.03 ,0.01,-0.008,-0.015, 0.022,0.005]
        g2List      =   [-0.015, 0.028,0.007,0.00, 0.020,-0.020,-0.005,0.010]
        num1A       =   0.  
        num2A       =   0.  
        for ig in range(nshear):
            g1 =    g1List[ig]
            g2 =    g2List[ig]
            for irot in range(nrot):
                prepend =   '-id%d-g%d-r%d' %(index,ig,irot)
                self.log.info('index: %d, shear: %d, rot: %d' %(index,ig,irot))
                inFname =   'src%s.fits' %(prepend)
                inFname =   os.path.join(inputDir,inFname)
                if not os.path.exists(inFname):
                    self.log.info('cannot find the input file%s' %prepend)
                    return
                src     =   astTab.Table.read(inFname)
                src1,src2   =   self.keepUnique(src,cat,irot,g1,g2)
                num1    =   len(src1)
                num2    =   len(src2)
                self.log.info('find gal1: %s, gal2: %s' %(num1,num2))
                num1A+=num1
                num2A+=num2
        numTT   =   50.*50.*nrot*nshear 
        return num1/numTT,num2A/numTT 



    def keepUnique(self,src,cat,irot,g1,g2):
        ngrid       =   80
        nx          =   50
        dist        =   cat['dist']
        angle       =   np.pi/4.*irot
        xg1,yg1,xg2,yg2 =   anaUtil.getPositions(dist,angle,g1,g2)
        minDist     =   min(dist/2./0.168,5.)
        src['ipos'] =   (src['base_SdssCentroid_y']//ngrid)*nx +(src['base_SdssCentroid_x']//ngrid).astype(np.int)
        src['xTcent']=  src['base_SdssCentroid_x']%ngrid-ngrid/2.+0.5
        src['yTcent']=  src['base_SdssCentroid_y']%ngrid-ngrid/2.+0.5
        # First, keep only detections that are the closest to galaxy1 
        src['disG1'] =  np.sqrt((src['xTcent']-xg1)**2.+(src['yTcent']-yg1)**2.)
        # Get sorted index by grid index and grid distance
        # Get the galaxies which is the closest to galaxy1 in the postage stamp
        inds        =   np.lexsort([src['disG1'],src['ipos']])
        inds_unique =   np.unique(src['ipos'][inds],return_index=True)[1]
        srcG1       =   src[inds[inds_unique]]
        srcG1       =   srcG1[(srcG1['disG1']<minDist)]
        srcG1       =   srcG1[np.all(~np.isnan(srcG1['fpfs_moments']),axis=1)]
        # Second, keep only detections that are the closest to galaxy2
        src['disG2'] =  np.sqrt((src['xTcent']-xg2)**2.+(src['yTcent']-yg2)**2.)
        # Get sorted index by grid index and grid distance
        # Get the galaxies which is the closest to galaxy1 in the postage stamp
        inds        =   np.lexsort([src['disG2'],src['ipos']])
        inds_unique =   np.unique(src['ipos'][inds],return_index=True)[1]
        srcG2       =   src[inds[inds_unique]]
        srcG2       =   srcG2[(srcG2['disG2']<minDist)]
        srcG2       =   srcG1[np.all(~np.isnan(srcG2['fpfs_moments']),axis=1)]
        return srcG1,srcG2



    @classmethod
    def writeConfig(self, butler, clobber=False, doBackup=False):
        pass

    def writeSchemas(self, butler, clobber=False, doBackup=False):
        pass

    def writeMetadata(self, ifield):
        pass

    def writeEupsVersions(self, butler, clobber=False, doBackup=False):
        pass
        

class anaUnDetectRateDriverConfig(pexConfig.Config):
    perGroup=   pexConfig.Field(dtype=int, default=100, doc = 'data per field')
    anaUnDetectRate = pexConfig.ConfigurableField(
        target = anaUnDetectRateTask,
        doc = "anaUnDetectRate task to run on multiple cores"
    )
    def setDefaults(self):
        pexConfig.Config.setDefaults(self)

class anaUnDetectRateRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        minGroup    =  parsedCmd.minGroup 
        maxGroup    =  parsedCmd.maxGroup 
        return [(ref, kwargs) for ref in range(minGroup,maxGroup)] 

def unpickle(factory, args, kwargs):
    """Unpickle something by calling a factory"""
    return factory(*args, **kwargs)

class anaUnDetectRateDriverTask(BatchPoolTask):
    ConfigClass = anaUnDetectRateDriverConfig
    RunnerClass = anaUnDetectRateRunner
    _DefaultName = "anaUnDetectRateDriver"
    
    def __reduce__(self):
        """Pickler"""
        return unpickle, (self.__class__, [], dict(config=self.config, name=self._name,
                parentTask=self._parentTask, log=self.log))

    def __init__(self,**kwargs):
        BatchPoolTask.__init__(self, **kwargs)
        self.makeSubtask("anaUnDetectRate")
    
    @abortOnError
    def run(self,Id):
        perGroup=   self.config.perGroup
        fMin    =   perGroup*Id
        fMax    =   perGroup*(Id+1)
        #Prepare the pool
        pool    =   Pool("anaUnDetectRate")
        pool.cacheClear()
        fieldList=  range(fMin,fMax)
        outs    =   pool.map(self.process,fieldList)
        names   =   ('num1_det','num2_det')
        tableO  =   astTab.Table(rows=outs,names=names)
        tableO.write('detectRate.csv')
        return
        
    def process(self,cache,ifield):
        self.anaUnDetectRate.run(ifield)
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
