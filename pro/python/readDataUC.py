#!/usr/bin/env python
#
# LSST Data Management System
# Copyright 20082014 LSST Corpoalphan.
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
import os
import anaUtil
import numpy as np
import astropy.table as astTab
from lsst.utils import getPackageDir
# for pipe task
import lsst.daf.base as dafBase
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
# lsst.afw...
import lsst.afw.math as afwMath
import lsst.afw.table as afwTable
import lsst.afw.image as afwImg
import lsst.afw.detection as afwDet
import lsst.afw.geom as afwGeom
import lsst.afw.coord as afwCoord
from lsst.meas.algorithms import SourceDetectionTask
from lsst.meas.deblender import SourceDeblendTask
from lsst.meas.base import SingleFrameMeasurementTask,CatalogCalculationTask

class readDataSimConfig(pexConfig.Config):
    "config"
    expPrefix   = pexConfig.Field(
        dtype=str, 
        default="expSim", 
        doc="prefix of input exposure"
    )
    srcPrefix   = pexConfig.Field(
        dtype=str, 
        default="outcomeHSM", 
        doc="prefiex of output src"
    )
    rootDir     = pexConfig.Field(
        dtype=str, 
        default="./rgc/", 
        doc="Root Diectory"
    )
    noiDir      = pexConfig.Field(
        dtype=str, 
        default="./corPre/", 
        doc="Diectory of noise correlation"
    )
    doAddFP   = pexConfig.Field(
        dtype=bool, 
        default=False, 
        doc="Whether add footprint",
    )
    doDeblend   = pexConfig.Field(
        dtype=bool, 
        default=True, 
        doc="Whether do deblending",
    )
    doWrite   = pexConfig.Field(
        dtype=bool, 
        default=True, 
        doc="Whether write outcome",
    )
    detection = pexConfig.ConfigurableField(
        target = SourceDetectionTask,
        doc = "Detect sources"
    )
    deblend = pexConfig.ConfigurableField(
        target = SourceDeblendTask,
        doc = "Split blended source into their components"
    )
    measurement = pexConfig.ConfigurableField(
        target = SingleFrameMeasurementTask,
        doc = "Measure sources"
    )
    catalogCalculation = pexConfig.ConfigurableField(
        target = CatalogCalculationTask,
        doc = "Subtask to run catalogCalculation plugins on catalog"
    )
    
    def setDefaults(self):
        pexConfig.Config.setDefaults(self)
        self.detection.thresholdValue   =   5.0
        self.detection.isotropicGrow    =   True
        self.detection.reEstimateBackground=False
        self.deblend.propagateAllPeaks  =   True 
        self.deblend.maxFootprintArea   =   64*64
        self.deblend.maxFootprintSize   =   64
        self.measurement.load(os.path.join(getPackageDir("obs_subaru"), "config", "hsm.py"))
        self.load(os.path.join(getPackageDir("obs_subaru"), "config", "cmodel.py"))

class readDataSimTask(pipeBase.CmdLineTask):
    ConfigClass = readDataSimConfig
    _DefaultName = "readDataSim"
    def __init__(self,schema,**kwargs):
        pipeBase.CmdLineTask.__init__(self, **kwargs)
        self.schema =   schema 
        self.algMetadata = dafBase.PropertyList()
        self.makeSubtask('detection', schema=self.schema)
        self.makeSubtask("deblend", schema=self.schema)
        self.makeSubtask('measurement', schema=self.schema, algMetadata=self.algMetadata)
        self.makeSubtask('catalogCalculation', schema=self.schema)
        if self.config.doAddFP:
            self.schema.addField("detected", type=np.int32,doc="wheter galaxy is detected by hscpipe")
            self.schema.addField("far2Cent", type=np.int32,doc="wheter galaxy is too far from the center of postage stamp")
        self.xT1=   0.  #x for first galaxy
        self.yT1=   0.  #y for first galaxy
        self.xT2=   0.  #x for second galaxy
        self.yT2=   0.  #y for second galaxy
        self.Dthre= 5.  #the minimumum distance to the input center 
                        #should be smaller than it
        
    def readData(self,prepend):
        index   =   eval(prepend[prepend.find('-id')+3:prepend.find('-g')]) 
        dist    =   astTab.Table.read('./catPre/control_cat.csv')[index]['dist']
        self.Dthre=  min(dist/0.168,5)
        irot    =   eval(prepend[prepend.find('-r')+2:]) 
        ang     =   np.pi/4.*irot
        g1List  =   [-0.02 ,-0.025,0.03 ,0.01,-0.008,-0.015, 0.022,0.005]
        g2List  =   [-0.015, 0.028,0.007,0.00, 0.020,-0.020,-0.005,0.010]
        ig      =   eval(prepend[prepend.find('-g')+2:prepend.find('-r')]) 
        g1      =   g1List[ig]
        g2      =   g2List[ig]
        self.xT1,self.yT1,self.xT2,self.yT2=anaUtil.getPositions(dist,ang,g1,g2)
        self.xT1/=0.168;self.yT1/=0.168;self.xT2/=0.168;self.yT2/=0.168
        #Read galaxy exposure
        expDir  =   os.path.join(self.config.rootDir,self.config.expPrefix)
        expfname=   'image%s.fits' %(prepend)
        expfname=   os.path.join(expDir,expfname)
        if not os.path.exists(expfname):
            self.log.info("cannot find the exposure")
            return None
        exposure=   afwImg.ExposureF.readFits(expfname)
        exposure.getMask().getArray()[:,:]=0
        if not exposure.hasPsf():
            self.log.info("exposure doesnot have PSF")
            return None
        
        #Read sources
        sourceDir   =   os.path.join(self.config.rootDir,self.config.srcPrefix)
        sourceFname =   os.path.join(sourceDir,'src%s.fits'%(prepend))
        if os.path.exists(sourceFname):
            sources =  afwTable.SourceCatalog.readFits(sourceFname) 
        else:
            sources=    self.measureSource(exposure,expfname)
            if sources is None:
                self.log.info('Cannot read sources')
                return None
            if self.config.doWrite:
                sources.writeFits(sourceFname)
        
        #Read power function of Poisson noise
        inPowsName  =   os.path.join(self.config.noiDir,'noiPows.npy')
        if not os.path.exists(inPowsName):
            self.log.info("cannot find power functions for noise")
            return None
        noiPows     =   np.load(inPowsName)
        
        return pipeBase.Struct(
            nImage  =   10*np.ones((exposure.getHeight(),exposure.getWidth()),dtype=int),
            faiPow  =   None,
            exposure=   exposure,
            sources=    sources,
            noiPows =   noiPows)
    


    def addFootPrint(self,exposure,sources):
        imgData =   exposure.getMaskedImage().getImage().getArray()
        #For footprints
        npkL=   [] # number of peaks
        xA1 =   0.;yA1 =   0.
        xA2 =   0.;yA2 =   0.
        numNpkE2=   0.
        for ss in sources:
            fp  =   ss.getFootprint()
            pks =   fp.peaks
            npk =   len(pks)
            if npk>=2:
                #Minimum dis1 and minimum dis2
                minD1   =   40.;minD2   =   40.;
                xcMin1  =   0.; ycMin1  =   0.;
                xcMin2  =   0.; ycMin2  =   0.;
                idp1    =   10; idp2    =   10;
                for ip,pk in enumerate(pks):
                    x   =   pk.getFx()
                    y   =   pk.getFy()
                    xc  =   x%80-39.5
                    yc  =   y%80-39.5
                    dis1=   np.sqrt((xc-self.xT1)**2.+(yc-self.yT1)**2.)
                    if dis1<minD1:
                        minD1=  dis1
                        idp1 =  ip
                        xcMin1=xc
                        ycMin1=yc
                    dis2=   np.sqrt((xc-self.xT2)**2.+(yc-self.yT2)**2.)
                    if dis2<minD2:
                        minD2=  dis2
                        idp2 =  ip
                        xcMin2= xc
                        ycMin2= yc
                if idp1==idp2 or minD1>self.Dthre or minD2>self.Dthre:
                    #idp1==idp2 means that the peat closest to c1 
                    #is the samme peak closest to c2, means one detection only
                    #minD1>self.Dthre or minD2>self.Dthre means false detection
                    npk=1
                else:
                    xA1+=xcMin1;yA1+=ycMin1
                    xA2+=xcMin2;yA2+=ycMin2
                    numNpkE2+=1.
            npkL.append(npk)
        if numNpkE2>10:
            xA1/=numNpkE2;yA1/=numNpkE2
            xA2/=numNpkE2;yA2/=numNpkE2
        else:
            xA1 =   self.xT1; yA1   =   self.yT1
            xA2 =   self.xT2; yA2   =   self.yT2

        for iss,ss in enumerate(sources):
            ss['detected']  =   1
            fp  =   ss.getFootprint()
            cFP =   fp.getCentroid()
            xc  =   cFP.getX()%80-39.5
            yc  =   cFP.getY()%80-39.5
            r   =   np.sqrt(xc**2.+yc**2.)
            if r>=8.:
                ss['far2Cent']  =   1
                ss['detected']  =   0
                fp.peaks.clear()
                fp.addPeak(xc,yc,1.)
                continue
            ss['far2Cent']  =   0
            if npkL[iss]==1:
                ss['detected']  =   0
            if npkL[iss]!=2:
                fp.peaks.clear()
                xSC =   cFP.getX()//80*80+39.5
                ySC =   cFP.getY()//80*80+39.5
                xPA1=   xSC+xA1;yPA1=   ySC+yA1
                xPA2=   xSC+xA2;yPA2=   ySC+yA2
                xPI1=   int(xPA1+0.5);yPI1=   int(yPA1+0.5)
                xPI2=   int(xPA2+0.5);yPI2=   int(yPA2+0.5)
                fp.addPeak(xPA1,yPA1,imgData[yPI1,xPI1])
                fp.addPeak(xPA2,yPA2,imgData[yPI2,xPI2])
        return
    
    def measureSource(self,exposure,expfname):
        table = afwTable.SourceTable.make(self.schema)
        sources = afwTable.SourceCatalog(table)
        table.setMetadata(self.algMetadata)
        detRes  = self.detection.run(table=table, exposure=exposure, doSmooth=True)
        sources = detRes.sources
        if self.config.doAddFP:
            #add undetected sources
            self.addFootPrint(exposure,sources)
        if self.config.doDeblend:
            # do deblending
            self.deblend.run(exposure=exposure,sources=sources)
        # do measurement
        self.measurement.run(measCat=sources,exposure=exposure)
        # measurement on the catalog level
        self.catalogCalculation.run(sources)
        exposure.writeFits(expfname)
        return sources
