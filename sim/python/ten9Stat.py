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
'''
Estimate shear from (<10^9) galaxies
Used to reply the referee
'''

import os
import galsim
import numpy as np
import astropy.io.fits as pyfits
from fpsBase import fpsBaseTask

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
import lsst.meas.base as meaBase
import lsst.meas.algorithms as meaAlg
from lsst.meas.algorithms import SourceDetectionTask
from lsst.meas.base import SingleFrameMeasurementTask

from lsst.pipe.base import TaskRunner
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, abortOnError


class ten9StatConfig(pexConfig.Config):
    "config"
    
    def setDefaults(self):
        pexConfig.Config.setDefaults(self)

class ten9StatTask(pipeBase.CmdLineTask):
    _DefaultName= "ten9Stat"
    ConfigClass = ten9StatConfig
    def __init__(self,**kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)
    
    @pipeBase.timeMethod
    def run(self,ifield):
        print 'processing field %s' %(ifield)
        # preparation
        minSnr  =   0.
        g1e     =   0.02; g2e   =   0.
        mName   =   'fps_momentsG'
        # load data
        iout    =   ifield/10000
        outputdir=  './out%d/' %(iout)
        catOutput=  outputdir+'src-%04d.fits'%(ifield)
        if not os.path.exists(catOutput):
            print catOutput+' does not exist'
            return None
        try: 
            data    =   pyfits.getdata(catOutput)        
        except:
            os.popen('rm '+catOutput)
            return None
        CBase   =   np.load('./CBase.npy')
        # get the position
        x       =   data['base_SdssCentroid_x']
        y       =   data['base_SdssCentroid_y']
        ngrid   =   48
        dx      =   abs(x%ngrid-ngrid/2)
        dy      =   abs(y%ngrid-ngrid/2)
        mask    =   np.ones(len(data)).astype(bool)
        mask    =   mask&(np.all(~np.isnan(data[mName]),axis=1))
        mask    =   mask&(dx**2.+dy**2.<100)&(x>ngrid)&(x<21*ngrid)&(y>ngrid)&(y<21*ngrid)
        data    =   data[mask]
        cRange  =   [4.]#np.arange(0.6,4.6,0.4)#
        nCR     =   len(cRange)
        msRange =   np.arange(10.,100.,10.)#np.arange(0,18,3)#
        nmR     =   len(msRange)
        # the output data
        dataO   =   np.zeros((nCR,nmR,6))
        for ic,cRatio in enumerate(cRange):
            const   =   CBase*cRatio
            # get proper snr
            """
            snr     =   data[mName][:,0]/(data[mName][:,0]+const)
            e1      =   data[mName][:,1]/(data[mName][:,0]+const)
            e2      =   data[mName][:,2]/(data[mName][:,0]+const)
            snr     =   snr-g1e*np.sqrt(2.)*e1*(1-snr)-g2e*np.sqrt(2.)*e2*(1-snr)
            snr     =   snr*100.
            """
            snr     =   data[mName][:,0]
            snr     =   data['snrI']
            for ims,maxSnr in enumerate(msRange):
                # mask according to the snr
                mask2   =   np.ones(len(data)).astype(bool)
                mask2   =   mask2&(snr<maxSnr)&(snr>=minSnr)
                data2   =   data[mask2]
                # shear measurement
                moments =   data2[mName]
                weight  =   moments[:,0]+const
                e1      =   -moments[:,1]/weight
                e2      =   -moments[:,2]/weight
                R1      =   1./np.sqrt(2.)*(moments[:,0]-moments[:,3])/weight+np.sqrt(2)*(moments[:,1]/weight)**2.
                R2      =   1./np.sqrt(2.)*(moments[:,0]-moments[:,3])/weight+np.sqrt(2)*(moments[:,2]/weight)**2.
                R       =   (R1+R2)/2.
                e1A     =   sum(e1)
                e2A     =   sum(e2)
                e1Sq    =   sum(e1**2.)
                e2Sq    =   sum(e2**2.)
                RA      =   sum(R)
                num     =   len(R)
                dataO[ic,ims,:]   =   np.array([RA,e1A,e2A,e1Sq,e2Sq,num],dtype='float64')
        return dataO

    @classmethod
    def _makeArgumentParser(cls):
        """
        Create an argument parser
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

class ten9StatBatchConfig(pexConfig.Config):
    ten9Stat = pexConfig.ConfigurableField(
        target = ten9StatTask,
        doc = "ten9Stat task to run on multiple cores"
    )
    
class ten9StatRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        refList =   range(100)
        refList.append(-1)
        return [(ref, kwargs) for ref in refList] 

def unpickle(factory, args, kwargs):
    """Unpickle something by calling a factory"""
    return factory(*args, **kwargs)

class ten9StatBatchTask(BatchPoolTask):
    ConfigClass = ten9StatBatchConfig
    RunnerClass = ten9StatRunner
    _DefaultName = "ten9StatBatch"

    def __reduce__(self):
        """Pickler"""
        return unpickle, (self.__class__, [], dict(config=self.config, name=self._name,
                parentTask=self._parentTask, log=self.log))

    def __init__(self,**kwargs):
        BatchPoolTask.__init__(self, **kwargs)
        self.makeSubtask("ten9Stat")
    
    @pipeBase.timeMethod
    def run(self,Id):
        fileName=   'outcomeBDKIntHigh5'
        if Id  >=   0:
            outFname=  './%s/out%d.npy' %(fileName,Id)
            if os.path.exists(outFname):
                return
            fMin    =   Id*10000
            fMax    =   fMin+10000
            #Prepare the pool
            pool    =   Pool("ten9Stat")
            pool.cacheClear()
            fieldList=  range(fMin,fMax)
            resList =   pool.map(self.process,fieldList)
            resList =   [x for x in resList if x is not None]
            resAll  =   np.zeros((1,9,6))
            for res in resList:
                resAll+=res
            np.save(outFname,resAll)
        if Id   <   0:
            lsout   =   os.popen('ls ./%s/ | grep .npy |grep out' %(fileName)) 
            files   =   lsout.readlines()
            finAll  =   np.zeros((1,9,6))
            for ff in files:
                final=  np.load('./%s/' %(fileName) +ff[:-1])
                finAll+=final
            np.save('./%s/eRes.npy' %(fileName),finAll )
        return
        
    def process(self,cache,ifield):
        return self.ten9Stat.run(ifield)

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
