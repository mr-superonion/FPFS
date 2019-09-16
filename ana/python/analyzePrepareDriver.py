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

# lsst Tasks
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base import TaskRunner
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, abortOnError

class analyzePrepareConfig(pexConfig.Config):
    "config"
    rootDir     = pexConfig.Field(
        dtype=str, 
        default="cgc-control-2gal/", 
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
        catPreName  =   os.path.join('catPre','control_cat.csv')
        assert os.path.exists(catPreName),'cannt find the preCat'
        cat         =   ascii.read(catPreName)[index]
        inputDir    =   os.path.join(self.config.rootDir,'outcomeFPFS')
        if not os.path.exists(inputDir):
            self.log.info('cannot find the input directory')
            return
        #Get the C
        cFname  =   'CBase.npy'
        cRatio  =   4. 
        const   =   (np.load(cFname))*cRatio
        if min(cat['flux1'],cat['flux2'])/cat['varNoi']<500.:
            nrot    =   8
        else:
            nrot    =   4
        nshear  =   8
        rows    =   []
        for ig in range(nshear):
            srcAll1 =   []
            minN1   =   2500
            srcAll2 =   []
            minN2   =   2500
            for irot in range(nrot):
                prepend =   '-id%d-g%d-r%d' %(index,ig,irot)
                self.log.info('index: %d, shear: %d, rot: %d' %(index,ig,irot))
                inFname =   'src%s.fits' %(prepend)
                inFname =   os.path.join(inputDir,inFname)
                if not os.path.exists(inFname):
                    self.log.info('cannot find the input file%s' %prepend)
                    return
                src     =   astTab.Table.read(inFname)
                #src    =   self.getNeibourInfo(src)
                src1,src2   =   self.keepUnique(src)
                num1    =   len(src1)
                if num1< minN1:
                    minN1=num1
                srcAll1.append(src1)
                num2    =   len(src2)
                if num2< minN2:
                    minN2=num2
                srcAll2.append(src2)
                self.log.info('%s,%s' %(minN1,minN2))
            srcF1 =   []
            srcF2 =   []
            for irot in range(nrot):
                srcF1.append(srcAll1[irot][:minN1])
                srcF2.append(srcAll2[irot][:minN2])
            #for the first galaxy
            del srcAll1
            srcF1  =   astTab.vstack(srcF1)
            row1   =   self.measureShear(srcF1,const)
            rows1.append(row1)
            #for the second galaxy
            del srcAll2
            srcF2  =   astTab.vstack(srcF2)
            row2   =   self.measureShear(srcF2,const)
            rows2.append(row2)
            #srcAll.write('src-%s-%s.fits' %(index,ig))
        return self.getMC(rows1)+self.getMC(rows2)


    def getMC(self,rows):
        names   =   ['g1e','g1err','g2e','g2err']
        tableO  =   astTab.Table(rows=rows,names=names)
        g1List  =   np.array([-0.02 ,-0.025,0.03 ,0.01,-0.008,-0.015, 0.022,0.005])
        g2List  =   np.array([-0.015, 0.028,0.007,0.00, 0.020,-0.020,-0.005,0.010])
        tableO['g1']=g1List
        tableO['g2']=g2List
        tableO.write('index%s.csv' %index)
        #Determine biases
        w1      =   1./tableO['g1err']
        w2      =   1./tableO['g2err']
        [m1,c1],cov1=np.polyfit(tableO['g1'],tableO['g1e'],1,w=w1,cov=True)
        [m2,c2],cov2=np.polyfit(tableO['g2'],tableO['g2e'],1,w=w2,cov=True)
        erm1=np.sqrt(cov1[0,0]);erc1=np.sqrt(cov1[1,1])
        erm2=np.sqrt(cov2[0,0]);erc2=np.sqrt(cov2[1,1])
        m1-=1;m2-=1
        print(m1,m2,c1,c2)
        print(erm1,erm2,erc1,erc2)
        return m1,m2,c1,c2,erm1,erm2,erc1,erc2


    def getPositions(self,cat,irot,ngrid,nx):
        #Get the input positions of galaxy1 and galaxy2
        dist        =   cat['dist']/2.
        angle       =   np.pi/4.*irot #a little bit wierd
        xg1         =   np.cos(angle)*dist
        yg1         =   np.sin(angle)*dist
        xg2         =   -np.cos(angle)*dist
        yg2         =   -np.sin(angle)*dist
        return xg1,yg1,xg2,yg2


    def keepUnique(self,src,cat,irot):
        ngrid       =   80
        nx          =   50
        xg1,yg1,xg2,yg2 =   self.getPositions(cat,irot,ngrid,nx)
        minDist     =   min(cat['dist']/2./0.168,5.)
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
        g1,g1err=   catStat.shearAverage(R1,e1) 
        g2,g2err=   catStat.shearAverage(R2,e2)
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
        fMin    =   perGroup*Id
        fMax    =   perGroup*(Id+1)
        #Prepare the pool
        pool    =   Pool("analyzePrepare")
        pool.cacheClear()
        fieldList=  range(fMin,fMax)
        outs    =   pool.map(self.process,fieldList)
        names   =   ('g1_m1','g1_m2','g1_c1','g1_c2','g1_erm1','g1_erm2','g1_erc1','g1_erc2')
        names   +=  ('g2_m1','g2_m2','g2_c1','g2_c2','g2_erm1','g2_erm2','g2_erc1','g2_erc2')
        tableO  =   astTab.Table(rows=outs,names=names)
        tableO.write('outcome.csv')
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
