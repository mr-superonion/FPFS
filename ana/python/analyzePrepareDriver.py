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
import k3match
import catStat
import numpy as np

import astropy.table as astTab

# lsst Tasks
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base import TaskRunner
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, abortOnError

import lsst.obs.subaru.filterFraction

class analyzePrepareConfig(pexConfig.Config):
    "config"
    rootDir     = pexConfig.Field(
        dtype=str, 
        default="rgc/fwhm4_var4/", 
        doc="Root Diectory"
    )
    def setDefaults(self):
        pexConfig.Config.setDefaults(self)

class analyzePrepareTask(pipeBase.CmdLineTask):
    _DefaultName = "analyzePrepare"
    ConfigClass = analyzePrepareConfig
    def __init__(self,**kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)
        
        
    @pipeBase.timeMethod
    def run(self,index):
        inputDir   =   os.path.join(self.config.rootDir,'outcomeFPFS')
        if not os.path.exists(inputDir):
            self.log.info('cannot find the input directory')
            return
        cFname  =   'CBase.npy'
        cRatio  =   4.
        const   =   (np.load(cFname))*cRatio
        rows    =   []
        for ig in range(8):
            srcAll  =   []
            minNum  =   2500
            for irot in range(4):
                prepend =   '-id%d-g%d-r%d' %(index,ig,irot)
                self.log.info('index: %d, shear: %d, rot: %d' %(index,ig,irot))
                inFname =   'src%s.fits' %(prepend)
                inFname =   os.path.join(inputDir,inFname)
                if not os.path.exists(inFname):
                    self.log.info('cannot find the input file%s' %prepend)
                    continue
                src     =   astTab.Table.read(inFname)
                self.log.info('%s' %len(src))
                #src     =   self.getNeibourInfo(src)
                #src     =   self.keepUnique(src)
                maskG       =   np.all(~np.isnan(src['fpfs_moments']),axis=1)
                src         =   src[maskG]
                num     =   len(src)
                self.log.info('%s' %num)
                if num< minNum:
                    minNum=num
                srcAll.append(src)
            srcAll2 =   []
            for irot in range(4):
                srcAll2.append(srcAll[irot][:minNum])
            srcAll  =   astTab.vstack(srcAll2)
            del srcAll2
            srcAll.write('src-%s-%s.fits' %(index,ig))
            row     =   self.measureShear(srcAll,const)
            rows.append(row)
        names   =   ['g1e','g1err','g2e','g2err']
        tableO  =   astTab.Table(rows=rows,names=names)
        g1List  =   np.array([-0.02 ,-0.025,0.03 ,0.01,-0.008,-0.015, 0.022,0.005])
        g2List  =   np.array([-0.015, 0.028,0.007,0.00, 0.020,-0.020,-0.005,0.010])
        tableO['g1']=g1List
        tableO['g2']=g2List
        tableO.write('index%s.csv' %index)
        return

    def getNeibourInfo(self,src):
        x   =   src['base_SdssShape_x']
        y   =   src['base_SdssShape_y']
        z   =   np.zeros(len(x))
        id1,id2,dis =   k3match.cartesian(x,y,z,x,y,z,50.)
        src2    =   src[id2]
        inds    =   np.lexsort([dis, id1])
        inds_unique =   np.unique(id1[inds], return_index=True)[1]
        id1     =   id1[inds[inds_unique]]
        id2     =   id2[inds[inds_unique]]
        dis     =   dis[inds[inds_unique]]
        src[id1]['distance_neibor']=dis
        namesU  =   ['base_FootprintArea_value','base_CircularApertureFlux_3_0_fluxSigma','base_CircularApertureFlux_3_0_flux']
        for name in namesU:
            nameA=  name+'_neibor'
            src[id1][nameA]=src[id2][name]
        return src
        

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
        src         =   src[inds[inds_unique]]
        src         =   src[(src['centDist']<5.)]
        #we also mask out galaxies without measurement
        return src

    def measureShear(self,src,const):
        #Shapelet modes
        moments =   src['fpfs_moments']
        #Get weight
        weight  =   moments[:,0]+const
        #Ellipticity
        e1      =   -moments[:,1]/weight
        e2      =   -moments[:,2]/weight
        #Response factor 
        R1      =   1./np.sqrt(2.)*(moments[:,0]-moments[:,3])/weight+np.sqrt(2)*(e1**2.)
        R2      =   1./np.sqrt(2.)*(moments[:,0]-moments[:,3])/weight+np.sqrt(2)*(e2**2.)
        RA      =   (R1+R2)/2.
        g1,g1err=   catStat.shearAverage(RA,e1) 
        g2,g2err=   catStat.shearAverage(RA,e2)
        return g1,g1err,g2,g2err
        
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
        fMin    =   perGroup*Id+100
        fMax    =   perGroup*(Id+1)+100
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
